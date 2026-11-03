# FootballCV design spec (2026-11-02)

Built from real product screens, captured in `design/references/` (full pages plus crops):
Veo Analytics 2 (`va_*.jpg`), Veo Editor (`ve_player.jpg`), Soccerment SICS Atlas (`sa_*.jpg`).
Rule: every element on screen answers a question a coach or player actually asks. If nothing on the screen
answers to it, it goes.

## 1. Structure: one match, four jobs

v1 was one long page holding every chart. Veo and Soccerment both split a match by the job you came to do.

| Tab | The question it answers | Reference |
|---|---|---|
| Overview | "How did the match go?" Score bar, the few numbers that decide a match, tap any event to watch it | Veo Match Report: "Get an instant, high level overview. Tap any event to play the clip, or a player to dive into their stats." (`va_events.jpg`) |
| Video | "Show me." The match with the AI layers on it and a timeline of moments | Veo Editor timeline and Speedrun (`ve_player.jpg`), Soccerment "Straight to the clip" |
| Players | "How did each player do?" One card per player, every stat with a Play pill | Veo player card (`va_players.jpg`) |
| Tactics | "Why did it go that way?" Space, shape and passing pictures on a clean pitch | Veo Visualisations (`va_pitch.jpg`) |
| Ask | "Tell me in words." Answers grounded in this match's data | Veo Coach Assist, Soccerment "Ask, in plain language" (`sa_ask.jpg`) |

A tab only appears when its data exists. A camera operated futsal stream has no pitch mapping yet, so Tactics
and the distance and speed rows are hidden rather than shown empty or faked.

## 2. Colour: the pitch decides

| Token | Value | Why |
|---|---|---|
| ground | `#121812` | Olive black, the Veo family. Grass footage sits on it without the blue cast a navy ground gives video. |
| surface | `#1A211A` | Cards one step up; Veo cards are a few percent lighter than the page with a soft top glow, never a border colour change alone. |
| raised | `#242C24` | Pills, inputs, hover. |
| line | `rgba(255,255,255,.08)` | Hairlines only. |
| text / muted | `#F3F4EF` / `#98A096` | Warm white like Veo's, not blue white. |
| primary action | white pill, ink text | Veo's "Buy now", "Play", "Get started" are all white pills. One per view. |
| live | `#C7F04B` | The only brand accent: the live dot, the playhead, the spotlight. It means "now". |
| good / warning / bad | `#4ED17E` / `#F2B33D` / `#F2555A` | Only for meaning: open pass, contested, lost. |
| teams | from the shirts | Team colour is data, so it comes from the video, snapped to a clean value. |

## 3. Type

| Use | Font | Why |
|---|---|---|
| Interface and headings | Schibsted Grotesk 400 to 600 | Closest free match to Veo's Oldschool Grotesk: a calm grotesk, headings at medium weight, not bold. |
| Big numbers | Archivo condensed 600 to 700 | Veo's "116.5 km" and score digits are condensed and tall; numbers get their own voice. |
| Labels | Schibsted Grotesk 600, 11px, letter spaced capitals | Veo's "PASSING NETWORK", Soccerment's "01 · FILTER". |

## 4. Pitch pictures

Black pitch (`#0B100B`), 1.5 px white lines, rounded outer corners, as in Veo's passing network and heat map.
Heat ramp green, yellow, orange, red. Pass network: red lines weighted by count, white nodes with the track
number in a small black dot. Every pitch is clickable to video where a moment exists.

## 5. Components

- Match bar: crest dot in team colour, score in Archivo, centre "date · Recording #n". (Veo)
- Stat row: label left muted, value right, then a Play pill: white when there is a clip, dimmed when not. (Veo)
- Event list: time, icon, title, who; tap plays the clip. (Veo Match Report)
- Video controls: icon buttons, previous and next moment, back 5 s, forward 5 s, a "Key moments only" switch
  (our Speedrun: plays only the clips), one "Layers" menu instead of ten toggles. (Veo Editor)
- Player card: avatar circle with the track number, "Min. tracked", rows with Play pills, a single
  "Open player" footer. (Veo)
- Section label: `01 · OVERVIEW` style small capitals over a one line promise. (Soccerment)

## 6. Copy

Short outcome lines, Veo style, no hype words: "Every player, tracked from the video you already have."
"Tap any moment to watch it." Never claim what the data cannot back.
