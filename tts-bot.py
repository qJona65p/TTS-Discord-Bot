import discord
from discord import app_commands

from collections import defaultdict
import configparser
import asyncio
import emoji
import json
import os
import re

import atexit
import subprocess
import threading
import time
import urllib.error
import urllib.request

# ---------------------    CONFIGS    ---------------------

config = configparser.ConfigParser()
config.read("config.cfg")

BOT_TOKEN           = config["Bot"]["token"]
IDLE_TIMEOUT        = int(config["Bot"]["idle_time"])

ALLOWED_CHANNEL_ID  = int(config["TTS"]["channel"]) if config["TTS"]["channel"].strip() else None
DEFAULT_LANGUAGE    = config["TTS"]["default_lang"]
DEFAULT_SPEAKER     = config["TTS"]["default_sp"]
MEDIA_MSG           = config["TTS"]["media_msg"]

ADMIN_IDS           = list(int(i) for i in config["Admin"]["admin_ids"].split(","))
RESTRICT_VOICES     = list(i for i in config["Admin"]["restricted_voices"].split(","))
AUTHORIZED_USERS    = list(int(i) for i in config["Admin"]["authorized_users"].split(","))

# qwentts.cpp tts-server
SERVER_BIN          = config["TTS"].get("server_bin", "").strip()   # empty = server is started by you
TALKER_GGUF         = config["TTS"].get("talker", "models/qwen-talker-1.7b-customvoice-Q8_0.gguf")
CODEC_GGUF          = config["TTS"].get("codec", "models/qwen-tokenizer-12hz-Q8_0.gguf")
SERVER_PORT         = int(config["TTS"].get("port", "8080"))
UNLOAD_AFTER        = int(config["TTS"].get("unload_after", "0"))   # seconds idle before freeing VRAM, 0 = never

# -------------    QWEN3-TTS (qwentts.cpp)    -------------

# config.cfg uses short codes (es, en...), Qwen3-TTS wants language names
LANG_MAP = {
    "es": "spanish", "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese", "it": "italian",
}

LANG_LABELS = {
    "es": "Español", "en": "Inglés", "zh": "Chino", "ja": "Japonés", "ko": "Coreano",
    "de": "Alemán", "fr": "Francés", "ru": "Ruso", "pt": "Portugués", "it": "Italiano",
}

def normalize_lang(value: str):
    """'es', 'Spanish' or 'Español' -> 'es'. Returns None if it isn't a supported language."""
    v = (value or "").strip().lower()
    if v in LANG_MAP:
        return v
    for code, name in LANG_MAP.items():
        if v in (name.lower(), LANG_LABELS[code].lower()):
            return code
    return None

# Built-in speakers of the CustomVoice checkpoints
BUILTIN_SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"]

class QwenTTS:
    """
    Talks to a qwentts.cpp `tts-server` over HTTP. Keeps the `.speakers` / `.tts_to_file(...)`
    interface the rest of the bot uses. If `server_bin` is set, this class starts the server
    on demand and can stop it again when idle so the GPU memory is released.
    """
    ALIAS = "qwen3-tts-customvoice"
 
    def __init__(self, server_bin: str, talker: str, codec: str, port: int):
        self.server_bin = server_bin
        self.talker = talker
        self.codec = codec
        self.url = f"http://127.0.0.1:{port}"
        self.port = port
        self.speakers = list(BUILTIN_SPEAKERS)
        self.proc = None
        self.last_used = time.monotonic()
        self._lock = threading.Lock()       # one request at a time, also guards start/stop
        self._log = None
        atexit.register(self.stop)          # never leave a server holding VRAM behind
 
    def find_speaker(self, name: str):
        """Case-insensitive lookup, returns the canonical speaker name or None."""
        for sp in self.speakers:
            if sp.lower() == (name or "").strip().lower():
                return sp
        return None
 
    def _is_up(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.url}/v1/audio/voices", timeout=2)
            return True
        except urllib.error.HTTPError:
            return True   # it answered, so it is listening
        except Exception:
            return False
 
    def ensure_running(self, timeout: float = 120):
        if self._is_up():
            return
        if not self.server_bin:
            raise RuntimeError(f"tts-server is not reachable at {self.url} and 'server_bin' is not set in config.cfg")
 
        if self.proc is None or self.proc.poll() is not None:
            env = os.environ.copy()
            env.setdefault("GGML_BACKEND", "CUDA0")
            self._log = open("tts-server.log", "ab")
            self.proc = subprocess.Popen(
                [self.server_bin, "--model", self.talker, "--codec", self.codec,
                "--alias", self.ALIAS, "--port", str(self.port)],
                env=env, stdout=self._log, stderr=self._log,
            )
            print(f"[TTSBot] Starting tts-server (pid {self.proc.pid}) ...")
 
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("tts-server exited during startup, see tts-server.log")
            if self._is_up():
                print("[TTSBot] tts-server ready")
                return
            time.sleep(0.5)
        raise RuntimeError("tts-server did not become ready in time, see tts-server.log")
 
    def stop(self):
        proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._log:
            self._log.close()
            self._log = None
 
    def unload_if_idle(self, idle_seconds: int) -> bool:
        """Stop the server we started if nobody used it for `idle_seconds`. Returns True if stopped."""
        if not self.proc or not self._lock.acquire(blocking=False):
            return False   # nothing to stop, or a request is in flight
        try:
            if self.proc and time.monotonic() - self.last_used > idle_seconds:
                self.stop()
                return True
            return False
        finally:
            self._lock.release()
 
    def tts_to_file(self, text: str, speaker: str, language: str, file_path: str):
        body = json.dumps({
            "model": self.ALIAS,
            "input": text,
            "voice": speaker.lower(),
            "language": LANG_MAP.get(language.lower(), language),
            "response_format": "wav",   # errors are reported properly with wav, not with pcm
        }).encode("utf-8")
 
        with self._lock:
            self.last_used = time.monotonic()
            self.ensure_running()
            req = urllib.request.Request(
                f"{self.url}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    audio = resp.read()
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"tts-server HTTP {e.code}: {e.read()[:200]!r}") from e
            self.last_used = time.monotonic()
 
        if not audio:
            raise RuntimeError("tts-server returned empty audio")
        with open(file_path, "wb") as f:
            f.write(audio)
 
# ---------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

class TTSBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tts = None
        self.join_locks = {}            # guild_id -> asyncio.Lock
        self._voice_clients = {}        # guild_id -> voice_client
        self.user_cfg = {}              # user_id -> speaker_name
        self.banned_users = {}

        self.load_bans()
        self.load_user_configs()
        self.reload_replacements()

        self._intentional_disconnect = set()
        self.queues = defaultdict(asyncio.Queue)   # guild_id -> asyncio.Queue of tasks
        self.processing = defaultdict(bool)        # guild_id -> is currently playing
        self.idle_tasks = {}

        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.tts = QwenTTS(SERVER_BIN, TALKER_GGUF, CODEC_GGUF, SERVER_PORT)
        if UNLOAD_AFTER > 0:
            print(f"[TTSBot] Model loads on first use and is freed after {UNLOAD_AFTER}s idle")
            self._unload_task = asyncio.create_task(self._unload_watcher())
        else:
            print("[TTSBot] Loading Qwen3-TTS (qwentts.cpp) ...")
            await asyncio.to_thread(self.tts.ensure_running)

        await self.tree.sync()
        print("[TTSBot] Bot ready and slash commands synced.")

    async def _unload_watcher(self):
        """Frees the GPU memory when TTS hasn't been used for UNLOAD_AFTER seconds."""
        while True:
            await asyncio.sleep(30)
            if await asyncio.to_thread(self.tts.unload_if_idle, UNLOAD_AFTER):
                print("[TTSBot] tts-server stopped (idle), VRAM released")
 
    async def close(self):
        if self.tts:
            await asyncio.to_thread(self.tts.stop)
        await super().close()

    def load_user_configs(self):
        try:
            with open("user_configs.json", "r") as file:
                data = json.load(file)
                self.user_cfg = {}
                for k, v in data.items():
                    key = int(k) if isinstance(k, str) and k.isdigit() else k
                    # Old files stored just the voice name as a string
                    self.user_cfg[key] = {"voice": v} if isinstance(v, str) else dict(v)

                print("[TTSBot] Loaded user configs")
        except Exception as e:
            print(f"[TTSBot] Could not load user_configs.json: {e}")
    
    def dump_user_configs(self):
        with open("user_configs.json", "w") as f:
            json.dump(self.user_cfg, f, indent=4)
        print("[TTSBot] Saved user configs")

    def resolve_speaker(self, user_id: int) -> str:
        """User's saved voice -> configured default -> first available. Old XTTS names fall through."""
        if self.tts:
            saved = self.user_cfg.get(user_id, {}).get("voice")
            if saved and self.tts.find_speaker(saved):
                return self.tts.find_speaker(saved)
            return self.tts.find_speaker(DEFAULT_SPEAKER) or self.tts.speakers[0]
        return DEFAULT_SPEAKER

    def resolve_language(self, user_id: int) -> str:
        """User's saved language -> configured default_lang -> Spanish. Always returns a code like 'es'."""
        return (
            normalize_lang(self.user_cfg.get(user_id, {}).get("lang"))
            or normalize_lang(DEFAULT_LANGUAGE)
            or "es"
        )

    def reload_replacements(self, interaction: discord.Interaction=None):
        try:
            with open("replacements.json", "r", encoding="utf-8") as f:
                self.replacements = json.load(f)
            print("[TTSBot] Replacements loaded")
            return True
        except Exception as e:
            self.replacements = {"emojis":{}, "symbols":{}}
            print(f"[TTSBot] Could not load replacements.json: {e}")
            return False

    def reload_config(self, interaction: discord.Interaction):
        """Reload the config.cfg file and update runtime values"""
        try:
            new_config = configparser.ConfigParser()
            new_config.read("config.cfg")

            # Update the values that are actually in use at runtime
            global IDLE_TIMEOUT, ALLOWED_CHANNEL_ID, DEFAULT_LANGUAGE, DEFAULT_SPEAKER, MEDIA_MSG, ADMIN_IDS, RESTRICT_VOICES, AUTHORIZED_USERS

            IDLE_TIMEOUT        = int(new_config["Bot"]["idle_time"])
            ALLOWED_CHANNEL_ID  = int(new_config["TTS"]["channel"]) if new_config["TTS"]["channel"].strip() else None
            DEFAULT_LANGUAGE    = new_config["TTS"]["default_lang"]
            DEFAULT_SPEAKER     = new_config["TTS"]["default_sp"]
            MEDIA_MSG           = new_config["TTS"]["media_msg"]
            ADMIN_IDS           = list(int(i) for i in new_config["Admin"]["admin_ids"].split(","))
            RESTRICT_VOICES     = list(i for i in new_config["Admin"]["restricted_voices"].split(","))
            AUTHORIZED_USERS    = list(int(i) for i in new_config["Admin"]["authorized_users"].split(","))

            print(f"[TTSBot] ({interaction.user.name}) Reloaded configuration")
            return True

        except Exception as e:
            print(f"[TTSBot] ({interaction.user.name}) Failed to reload configuration: {e}")
            return False

    def load_bans(self):
        """Load persistent bans from file"""
        try:
            with open("bans.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.banned_users = {int(k): v for k, v in data.items()}
            print(f"[TTSBot] Loaded {len(self.banned_users)} banned users")
        except FileNotFoundError:
            self.banned_users = {}
        except Exception as e:
            print(f"[TTSBot] Error loading bans.json: {e}")
            self.banned_users = {}

    def dump_bans(self):
        """Save bans to file"""
        try:
            with open("bans.json", "w", encoding="utf-8") as f:
                json.dump(self.banned_users, f, indent=4, ensure_ascii=False)
            print("[TTSBot] Bans saved")
        except Exception as e:
            print(f"[TTSBot] Error saving bans: {e}")

    def is_banned(self, user_id: int) -> bool:
        """Check if user is banned or timed out"""
        if user_id not in self.banned_users:
            return False
        
        ban_info = self.banned_users[user_id]
        
        if isinstance(ban_info, dict) and "until" in ban_info:
            if ban_info["until"] < asyncio.get_event_loop().time():
                # Ban expired → remove it
                del self.banned_users[user_id]
                self.dump_bans()
                return False
            return True # Temporary ban (timeout)
        return True  # Permanent ban    

    async def join_voice(self, interaction: discord.Interaction=None, message: discord.Message=None):
        if message:
            if not message.author.voice or not message.author.voice.channel:
                return None
            channel = message.author.voice.channel
            guild_id = message.guild.id

            # not join if AFK channel
            if channel.id == message.guild._afk_channel_id:
                print(f"[TTSBot] Ignored AFK channel: {channel.name}")
                return None

        elif interaction:
            if not interaction.user.voice or not interaction.user.voice.channel:
                await interaction.followup.send("Debes unirte a un canal de voz primero.", ephemeral=True)
                return None

            channel = interaction.user.voice.channel
            guild_id = interaction.guild.id

            # not join if AFK channel
            if channel.id == interaction.guild._afk_channel_id:
                await interaction.followup.send("No puedo unirme al canal AFK.", ephemeral=True)
                print(f"[TTSBot] Ignored message from AFK channel: {channel.name}")
                return None
        else: 
            return None

        if guild_id not in self.join_locks:
            self.join_locks[guild_id] = asyncio.Lock()

        async with self.join_locks[guild_id]:
            return await self._join_voice_internal(channel, guild_id)

    async def _join_voice_internal(self, channel, guild_id: int):
        vc = self._voice_clients.get(guild_id)

        if vc and not vc.is_connected():
            await self.force_cleanup(guild_id)
            vc = None

        if vc and vc.is_connected():
            if vc.channel.id == channel.id:
                self.reset_idle_timer(guild_id)
                return vc
            else:
                # Move to different channel
                try:
                    if vc.is_playing():
                        vc.stop()
                    await self.force_cleanup(guild_id)
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"[Move Error] {e}")
                    await self.force_cleanup(guild_id)
                vc = None

        # === Fresh connection with retries ===
        for attempt in range(3):  # up to 3 attempts
            try:
                print(f"[TTSBot] Join attempt {attempt+1}/3")
                vc = await asyncio.wait_for(
                    channel.connect(),
                    timeout=6.0 if attempt == 0 else 4.0
                )
                self._voice_clients[guild_id] = vc
                print(f"[TTSBot] Joined voice channel: {channel.name}")
                self.reset_idle_timer(guild_id)
                return vc

            except asyncio.TimeoutError:
                print(f"[Join Timeout] Attempt {attempt+1}")
                await self.force_cleanup(guild_id)
                await asyncio.sleep(0.8 if attempt < 2 else 1.5)

            except discord.ClientException as e:  # Already connected
                print(f"[Join Error] Already connected: {e}")
                await self.force_cleanup(guild_id)
                await asyncio.sleep(0.7)

            except Exception as e:
                print(f"[Join Error] {type(e).__name__}: {e}")
                await self.force_cleanup(guild_id)
                await asyncio.sleep(1.0)

        print(f"[TTSBot] Failed to join voice after 3 attempts")
        return None

    def reset_idle_timer(self, guild_id: int):
        """Reset the idle disconnect timer"""
        # Cancel existing timer
        if guild_id in self.idle_tasks and not self.idle_tasks[guild_id].done():
            self.idle_tasks[guild_id].cancel()

        # Create new timer
        self.idle_tasks[guild_id] = asyncio.create_task(self._idle_disconnect(guild_id))
    
    async def _idle_disconnect(self, guild_id: int):
        """Background task that disconnects after IDLE_TIMEOUT seconds of inactivity"""
        try:
            await asyncio.sleep(IDLE_TIMEOUT)
            
            # If we reach here, no activity happened
            if guild_id in self._voice_clients and self._voice_clients[guild_id].is_connected():
                vc = self._voice_clients[guild_id]
                try:
                    await vc.disconnect()
                    print(f"[TTSBot] Auto-disconnected from guild {guild_id} due to inactivity")
                except Exception as e:
                    print(f"[TTSBot] Error while idle disconnecting: {e}")
                
                self._voice_clients.pop(guild_id, None)
                # Clear queue
                while not self.queues[guild_id].empty():
                    try:
                        self.queues[guild_id].get_nowait()
                        self.queues[guild_id].task_done()
                    except:
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TTSBot] Idle Timer Error: {e}")

    def preprocess_text(self, text:str, lang: str = "es"):
        """Clean text for better TTS"""
        if not text:
            return ""

        # Link removal
        pattern = r'(https?://)?([a-z0-9.-]+\.[a-z]{2,})([/\w.-]*)?(\?[^#\s]*)?(#[^\s]*)?'
        text = re.sub(pattern, "Link", text)

        # Handle Discord custom emojis first (<:name:id> or <a:name:id>)
        def replace_discord_emoji(match):
            full = match.group(0)
            # Extract the name (everything between the first : and the last :)
            name_match = re.search(r':([^:]+):', full)
            if name_match:
                name = name_match.group(1)
                return f"{name.replace('_', ' ')} "
            return " "

        text = re.sub(r'<a?:[^:]+:\d+>', replace_discord_emoji, text)
        section = self.replacements if lang == "es" else self.replacements.get(lang, {})
        emoji_mapping = section.get("emojis", {})
        symbol_mapping = section.get("symbols", {})

        # Convert emojis to descriptions
        def replace_unicode_emoji(emoji_char, data=None):
            if emoji_char in emoji_mapping:
                return emoji_mapping[emoji_char]
            try:
                name = emoji.demojize(emoji_char, language=lang)
            except Exception:
                name = emoji.demojize(emoji_char, language="en")  # language not covered by the emoji package
            return name.replace(":", " ").replace("_", " ").strip()

        text = emoji.replace_emoji(text, replace=replace_unicode_emoji)

        # Fix common symbols / punctuation
        for old, new in symbol_mapping.items():
            text = text.replace(old, new)

        return text

    async def _play_text(self, voice_client, text: str, speaker: str, language: str, interaction=None):
        """Internal method to generate and play one message"""

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            try:
                if not voice_client.is_connected():
                    print("[TTSBot] Voice client disconnected mid-playback, aborting.")
                    return  # Let process_queue handle the broken state

                temp_path = f"temp_tts_{voice_client.guild.id}.wav"
                
                await asyncio.to_thread(
                    self.tts.tts_to_file,
                    text=sentence.strip(),
                    speaker=speaker,
                    language=language,
                    file_path=temp_path
                )
                if not voice_client.is_connected():  # Check again after TTS generation
                    return

                voice_client.play(discord.FFmpegPCMAudio(temp_path))

                while voice_client.is_playing():
                    await asyncio.sleep(0.2)

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"[TTS Error] Playing: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)  # Clean up even on error
                if interaction and interaction.channel:
                    try:
                        await interaction.channel.send(f"Error generando audio: {str(e)[:100]}")
                    except:
                        pass
    
    async def process_tts_message(self, message: discord.Message):
        """Automatically speak any message sent in the monitored text channel"""
        if self.is_banned(message.author.id):
            try:
                await message.add_reaction("❌")
            except:
                pass
            return
        
        guild_id = message.guild.id

        # Get the user's current voice channel
        if not message.author.voice or not message.author.voice.channel:
            return

        # Join voice if needed
        vc = await self.join_voice(message=message)
        if not vc:
            return

        # Get speaker for this user (or default)
        speaker = self.resolve_speaker(message.author.id)
        lang = self.resolve_language(message.author.id)

        # Preprocess text + detect media
        clean_text = self.preprocess_text(message.content, lang)

        if message.attachments:
            clean_text += f" {MEDIA_MSG}"

        # Add to queue (text = message.content)
        await self.queues[guild_id].put((clean_text, speaker, lang, None))  # interaction=None for auto mode

        # Start queue processor
        asyncio.create_task(self.process_queue(guild_id))

        # React to the message so users know it's being processed
        try:
            await message.add_reaction("🎙️")
        except:
            pass
    
    async def process_queue(self, guild_id: int):
        """Background task that processes the queue for a guild"""
        if self.processing[guild_id]:
            return
        self.processing[guild_id] = True

        try:
            while True:
                task = await asyncio.wait_for(self.queues[guild_id].get(), timeout=30)
                text, speaker, lang, interaction = task

                vc = self._voice_clients.get(guild_id)
                if not vc or not vc.is_connected():
                    break

                await self._play_text(vc, text, speaker, lang, interaction)

                self.queues[guild_id].task_done()

                # Small delay between messages
                await asyncio.sleep(0.2)
        except TimeoutError:
            pass
        except Exception as e:
            print(f"[TTSBot] Queue Error: Guild {guild_id}: {e}")
        finally:
            self.processing[guild_id] = False

    async def voice_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for /setvoice - shows only real speakers"""
        if not self.tts or not self.tts.speakers:
            return []
        
        # Filter speakers that match what the user is typing
        current = current.lower()
        matching = [
            app_commands.Choice(name=speaker, value=speaker)
            for speaker in self.tts.speakers
            if current in speaker.lower()
        ]
        
        # Return up to 25 choices (Discord limit)
        return matching[:25]
    
    async def language_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for /setlang - matches the code, the Spanish name or the English name"""
        current = current.lower().strip()
        matching = [
            app_commands.Choice(name=f"{LANG_LABELS[code]} ({code})", value=code)
            for code, name in LANG_MAP.items()
            if not current or current in code or current in name.lower() or current in LANG_LABELS[code].lower()
        ]
        
        # Return up to 25 choices (Discord limit)
        return matching[:25]

    async def force_cleanup(self, guild_id: int):
        """Aggressive cleanup when state is broken"""
        print(f"[TTSBot] Force cleaning voice state for guild {guild_id}")

        self._intentional_disconnect.add(guild_id)

        vc = self._voice_clients.pop(guild_id, None)
        if vc:
            try:
                await vc.disconnect(force=True)
            except:
                pass

        # Clear discord internal tracking
        guild = self.get_guild(guild_id)
        if guild and guild.voice_client:
            if vc is None or guild.voice_client is vc:
                try:
                    await guild.voice_client.disconnect(force=True)
                except:
                    pass

        # Clear queue
        self.processing[guild_id] = False
        queue = self.queues[guild_id]
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except:
                break

        if guild_id in self.idle_tasks and not self.idle_tasks[guild_id].done():
            self.idle_tasks[guild_id].cancel()

        await asyncio.sleep(0.3)
        self._intentional_disconnect.discard(guild_id)
    
# ---------------------- BOT EVENTS ----------------------
client = TTSBot()

@client.event
async def on_ready():
    print(f"[TTSBot] Logged in as {client.user}")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Only react to messages in the specific TTS channel
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    # Process the message for TTS
    await client.process_tts_message(message)

@client.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    if member.id != client.user.id:
        return

    guild_id = None
    if before.channel:
        guild_id = before.channel.guild.id
    elif after.channel:
        guild_id = after.channel.guild.id

    if not guild_id:
        return

    if before.channel and not after.channel:
        # Skip if this disconnect was triggered by our own cleanup
        if guild_id in client._intentional_disconnect:
            print(f"[TTSBot] Ignoring intentional disconnect event for guild {guild_id}")
            return
        # Skip if we've already reconnected
        vc = client._voice_clients.get(guild_id)
        if vc and vc.is_connected():
            print(f"[TTSBot] Ignoring stale disconnect event — already reconnected")
            return

        print(f"[TTSBot] Bot disconnected from voice in guild {guild_id}")
        if guild_id not in client.join_locks:
            client.join_locks[guild_id] = asyncio.Lock()
        async with client.join_locks[guild_id]:
            await client.force_cleanup(guild_id)

# -------------------   COMMANDS   -------------------

@client.tree.command(name="tts", description="Un tts we, que esperabas")
@app_commands.describe(text="El texto que quieres que hable")
async def tts(interaction: discord.Interaction, text: str):
    if ALLOWED_CHANNEL_ID and interaction.channel_id != ALLOWED_CHANNEL_ID:
        await interaction.response.send_message("Este comando solo se puede usan en el canal de TTS.", ephemeral=True)
        return
    
    await interaction.response.defer()

    if client.is_banned(interaction.user.id):
        await interaction.followup.send("Baneao.", ephemeral=True)
        return

    vc = await client.join_voice(interaction=interaction)
    if not vc:
        return
    
    # Get user's chosen voice or default
    speaker = client.resolve_speaker(interaction.user.id)
    lang = client.resolve_language(interaction.user.id)

    clean_text = client.preprocess_text(text)

    # Add to queue
    await client.queues[interaction.guild.id].put((clean_text, speaker, lang, interaction))

    # Start queue processor if not running
    asyncio.create_task(client.process_queue(interaction.guild.id))

    await interaction.followup.send(f"**{speaker}** en cola ({client.queues[interaction.guild.id].qsize()} mensajes)", ephemeral=True)

@client.tree.command(name="setvoice", description="Selecciona la voz para el TTS")
@app_commands.describe(voice="Nombre de la voz (usa /voices para verlas)")
@app_commands.autocomplete(voice=client.voice_autocomplete)
async def setvoice(interaction: discord.Interaction, voice: str):
    if not client.tts or not client.tts.speakers:
        await interaction.response.send_message("El modelo aún no ha cargado las voces.", ephemeral=True)
        return
    
    if voice not in client.tts.speakers:
        await interaction.response.send_message(f"La voz **{voice}** no existe.\n", ephemeral=True)
        return

    voice_name = voice.strip()

    if voice_name in RESTRICT_VOICES and interaction.user.id != AUTHORIZED_USERS[RESTRICT_VOICES.index(voice_name)]: # Restriccion de voces
        await interaction.response.send_message(f"No autorizo.\n", ephemeral=True)
        return
    
    # Save option
    client.user_cfg.setdefault(interaction.user.id, {})["voice"] = voice_name
    client.dump_user_configs()
    await interaction.response.send_message(f"Tu voz ha sido cambiada a **{voice}**.", ephemeral=True)

@client.tree.command(name="setlang", description="Selecciona el idioma en el que habla tu voz")
@app_commands.describe(language="Idioma (empieza a escribir para ver las opciones)")
@app_commands.autocomplete(language=client.language_autocomplete)
async def setlang(interaction: discord.Interaction, language: str):
    code = normalize_lang(language)
    if not code:
        await interaction.response.send_message(f"El idioma **{language}** no existe.", ephemeral=True)
        return

    client.user_cfg.setdefault(interaction.user.id, {})["lang"] = code
    client.dump_user_configs()
    await interaction.response.send_message(f"Tu idioma ha sido cambiado a **{LANG_LABELS[code]}**.", ephemeral=True)

@client.tree.command(name="voices", description="Muestra las voces disponibles")
async def voices(interaction: discord.Interaction):
    if not client.tts or not client.tts.speakers:
        await interaction.response.send_message("Las voces aún no están cargadas.", ephemeral=True)
        return
    
    speakers_list = "\n".join([f"• {s}" for s in client.tts.speakers[:60]])
    embed = discord.Embed(
        title="Voces disponibles (Primeras 60)",
        description=speakers_list,
        color=0x00ff00
    )
    embed.set_footer(text=f"Ejemplo: /setvoice {DEFAULT_SPEAKER}")
    await interaction.response.send_message(embed=embed, ephemeral=True)
    
@client.tree.command(name="leave", description="Desconectar el bot del canal de voz")
async def leave(interaction: discord.Interaction):
    guild_id = interaction.guild.id
    if guild_id in client._voice_clients and client._voice_clients[guild_id].is_connected():
        await client._voice_clients[guild_id].disconnect()
        del client._voice_clients[guild_id]
        # Clear queue
        while not client.queues[guild_id].empty():
            try:
                client.queues[guild_id].get_nowait()
                client.queues[guild_id].task_done()
            except:
                break
        await interaction.response.send_message("Bot desconectado del canal de voz.", ephemeral=True)
    else:
        await interaction.response.send_message("El bot no está en ningún canal de voz.", ephemeral=True)

# -------------------  ADMIN COMMANDS   -------------------

@client.tree.command(name="ban", description="Bloquear el uso del bot a un usuario")
@app_commands.describe(user="Usuario a banear")
async def ban(interaction: discord.Interaction, user: discord.User):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return

    user_id = user.id
    client.banned_users[user_id] = "permanent"
    client.dump_bans()

    await interaction.response.send_message(f"Usuario **{user}** ({user_id}) ha sido baneado.", ephemeral=True)

@client.tree.command(name="unban", description="Desbloquear el uso del bot a un usuario")
@app_commands.describe(user="Usuario a desbanear")
async def unban(interaction: discord.Interaction, user: discord.User):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return
    
    user_id = user.id
    if user_id in client.banned_users:
        del client.banned_users[user_id]
        client.dump_bans()
        await interaction.response.send_message(f"Usuario **{user}** ha sido desbaneado.", ephemeral=True)
    else:
        await interaction.response.send_message(f"El usuario no estaba baneado.", ephemeral=True)

@client.tree.command(name="timeout", description="Bloquear el uso del bot a un usuario temporalmente")
@app_commands.describe(user="Usuario", minutes="Duración en minutos (default: 3)")
async def timeout(interaction: discord.Interaction, user: discord.User, minutes: int=3):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return
    
    user_id = user.id
    until = asyncio.get_event_loop().time() + (minutes * 60)

    client.banned_users[user_id] = {"until": until, "reason": f"Timeout de {minutes} minutos"}
    client.dump_bans()

    await interaction.response.send_message(
        f"Usuario **{user}** ha sido baneado por **{minutes} minutos**.", 
        ephemeral=True
    )
    
@client.tree.command(name="reload", description="Recargar configuraciones del bot. (Admins)")
@app_commands.choices(option=[
    app_commands.Choice(name="Configs", value="Configs"),
    app_commands.Choice(name="Replacements", value="Replacements"),
])
async def reload(interaction: discord.Interaction, option: str):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return

    if option == "Configs":
        await interaction.response.defer(ephemeral=True)
        success = client.reload_config(interaction)

        if success:
            await interaction.followup.send("Configuración recargada correctamente.", ephemeral=True)
        else:
            await interaction.followup.send("Error al recargar la configuración.", ephemeral=True)
    
    elif option == "Replacements":
        await interaction.response.defer(ephemeral=True)
        success = client.reload_replacements(interaction)

        if success:
            await interaction.followup.send("Replacements recargados correctamente.", ephemeral=True)
        else:
            await interaction.followup.send("Error al recargar los replacements.", ephemeral=True)

client.run(BOT_TOKEN)
