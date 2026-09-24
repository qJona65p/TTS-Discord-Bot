# Discord TTS Bot (Qwen3-TTS)

A Discord bot that reads text-channel messages aloud in your voice channel using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), run locally on your own GPU through [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) with an 8-bit quantized (Q8_0) model.

Each user can pick their own **voice** and **language**, and the model can be unloaded automatically when nobody is using it, so the GPU memory is free again.

> The bot's replies and slash-command descriptions are in Spanish.

## Features

- Speaks every message sent in a chosen text channel to the voice channel the author is in
- Per-user voice (`/setvoice`) and language (`/setlang`) with autocomplete, saved between restarts
- 9 built-in voices, 10 languages
- Per-server message queue, so messages are read in order
- Custom text cleanup: links, Discord emojis, Unicode emojis and symbols are turned into speakable text (see [Text replacements](#text-replacements))
- Optional idle unloading of the model to release VRAM
- Admin tools: ban, timeout, restricted voices, live config reload

## How it works

```
Discord message ──► tts-bot.py ──HTTP──► qwentts.cpp tts-server (Qwen3-TTS, CUDA) ──► WAV ──► Discord voice
```

The bot doesn't load the model itself. It talks to the `tts-server` from qwentts.cpp over a local HTTP API (`/v1/audio/speech`). If you set `server_bin` in the config, the bot starts that server when it's needed and stops it again after `unload_after` seconds without use.

## Requirements

- An NVIDIA GPU. The 1.7B Q8_0 model is about 2 GB of weights, plus overhead. Sets around ~4GB of VRAM usage.
- CUDA toolkit (needed to build qwentts.cpp)
- Python 3.10+
- `ffmpeg` and `libopus` (for Discord voice)
- A Discord bot application with a token

## Setup

### 1. Build qwentts.cpp

```bash
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
cd qwentts.cpp
./buildcuda.sh
```

### 2. Download the models

You need two files: the talker (the CustomVoice model with the named speakers) and the audio codec.

I've used this two
- `qwen-talker-1.7b-customvoice-Q8_0.gguf`
- `qwen-tokenizer-12hz-Q8_0.gguf`

### 3. Test the server on its own

```bash
GGML_BACKEND=CUDA0 ./qwentts.cpp/build/tts-server \
    --model models/qwen-talker-1.7b-customvoice-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --alias qwen3-tts-customvoice --port 8080
```

In another terminal:

```bash
curl -X POST localhost:8080/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-customvoice","input":"Hola, ¿qué tal?","voice":"aiden","language":"spanish","response_format":"wav"}' \
  -o out.wav
```

Play `out.wav`. If it sounds right, stop the server. The bot will start its own if you configure it to do so.

### 4. Install the bot's dependencies

```bash
sudo apt install ffmpeg libopus0
pip install -r requierements.txt
```

### 5. Create the Discord application

1. Create an application and a bot at the [Discord Developer Portal](https://discord.com/developers/applications) and copy the token.
2. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent**. Without it the bot can't read messages.
3. Invite the bot with the `bot` and `applications.commands` scopes and these permissions: View Channels, Add Reactions, Read Message History, Connect, Speak.

### 6. Configure

Edit `config.cfg` (see the [reference](#configuration-reference)). At minimum, set your token, the TTS channel, and the paths to the server and models.

### 7. Run

Run the bot from the folder that contains `models/` and `qwentts.cpp/` (the paths in the config are relative to where you launch it):

```bash
python tts-bot.py
```

## Configuration reference

```ini
[Bot]
token       = YOUR_DISCORD_BOT_TOKEN
idle_time   = 300

[TTS]
channel             = 123456789012345678
default_lang        = es
default_sp          = aiden
media_msg           = media
server_bin          = ./qwentts.cpp/build/tts-server
talker              = models/qwen-talker-1.7b-customvoice-Q8_0.gguf
codec               = models/qwen-tokenizer-12hz-Q8_0.gguf
port                = 8080
unload_after        = 600

[Admin]
admin_ids           = 111111111111111111,222222222222222222
restricted_voices   = eric,dylan
authorized_users    = 111111111111111111,222222222222222222
```

| Key | Description |
|---|---|
| `[Bot] token` | Discord bot token. |
| `[Bot] idle_time` | Seconds of inactivity before the bot leaves the voice channel. |
| `[TTS] channel` | ID of the text channel the bot reads from. Leave empty to read every channel it can see. When set, `/tts` also only works there. |
| `[TTS] default_lang` | Language for users who haven't picked one. Accepts a code (`es`). |
| `[TTS] default_sp` | Voice for users who haven't picked one. Must be one of the [built-in voices](#voices-and-languages). |
| `[TTS] media_msg` | Word spoken after a message that has attachments. |
| `[TTS] server_bin` | Path to the `tts-server` binary. If set, the bot starts and stops the server itself. Leave empty if you run the server yourself. |
| `[TTS] talker` / `codec` | Paths to the two model files. |
| `[TTS] port` | Port of the `tts-server`. |
| `[TTS] unload_after` | Seconds without use before the server is stopped to free VRAM. The next message starts it again, so that one is slower. `0` loads the model at startup and keeps it loaded. |
| `[Admin] admin_ids` | Discord user IDs allowed to use the admin commands. Comma-separated. |
| `[Admin] restricted_voices` | Voices only certain users may select. Comma-separated, lowercase, **no spaces**. |
| `[Admin] authorized_users` | The user allowed to use each restricted voice, matched **by position** with `restricted_voices` (first ID ↔ first voice, and so on). Must contain at least one valid ID or the bot won't start. |

The server settings (`server_bin`, `talker`, `codec`, `port`, `unload_after`) are only read at startup. Everything else can be reloaded with `/reload`.

## Commands

| Command | Who | What it does |
|---|---|---|
| `/tts <text>` | Everyone | Speaks the text in your voice channel. |
| `/setvoice <voice>` | Everyone | Choose your voice (autocomplete). |
| `/setlang <language>` | Everyone | Choose your language (autocomplete). |
| `/voices` | Everyone | Lists the available voices. |
| `/leave` | Everyone | Makes the bot leave the voice channel and clears the queue. |
| `/ban <user>` / `/unban <user>` | Admins | Block or unblock a user. |
| `/timeout <user> [minutes]` | Admins | Block a user temporarily (default 3 minutes). |
| `/reload <Configs\|Replacements>` | Admins | Reload `config.cfg` or `replacements.json` without restarting. |

Besides the slash commands, any message sent in the TTS channel by someone who is in a voice channel is read out automatically. The bot joins that user's channel (it ignores the server's AFK channel) and reacts to show the message was queued.

## Voices and languages

**Voices:** `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`

None of them is a native Spanish speaker. Qwen recommends each voice's native language for the best quality, so try them in your language before settling on one. `eric` and `dylan` are Mandarin dialect voices (Sichuan and Beijing) and use those dialects when the language is Chinese.

**Languages:** Español (`es`), Inglés (`en`), Chino (`zh`), Japonés (`ja`), Coreano (`ko`), Alemán (`de`), Francés (`fr`), Ruso (`ru`), Portugués (`pt`), Italiano (`it`)

## Text replacements

Before speaking, the bot cleans each message: links become "Link", custom Discord emojis are read by name, and Unicode emojis and symbols are turned into words.

`replacements.json` controls this. Its top-level `emojis` and `symbols` sections are the **Spanish** replacements (for example `?` → "pregunta"). Other languages can have their own section:

```json
{
    "emojis":  { "😂": "lagrimas de felicidad" },
    "symbols": { "?": " pregunta " },

    "en": {
        "emojis":  { "😂": "tears of joy" },
        "symbols": { "...": " dot dot dot " }
    }
}
```

For a language without a section, emojis are named in that language by the [`emoji`](https://pypi.org/project/emoji/) package (falling back to English) and punctuation is left as it is.

## Files the bot creates

| File | Purpose |
|---|---|
| `user_configs.json` | Each user's saved voice and language. |
| `bans.json` | Banned and timed-out users. |
| `tts-server.log` | Output of the `tts-server` the bot started. Check it first when something goes wrong. |
| `temp_tts_<guild>.wav` | Temporary audio while a sentence is playing. |

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba Cloud
- [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) and the [GGUF models](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) by ServeurpersoCom
- [discord.py](https://github.com/Rapptz/discord.py)