# spinf agent skills

Agent skills for building with [spinf](https://spinf.com): many questions about a piece of content in one call, a
probability for each answer you allow, no generated text. The same API runs **hosted** at `api.spinf.com` and
**self-hosted** with [spinf-serve](https://github.com/spinfinc/spinf-serve), the open-source engine; the skills cover both.

| Skill | For | Purpose |
|---|---|---|
| [spinf](skills/spinf/SKILL.md) | code that calls the API | Request rules, reading `p` / `floor_p` / `residual`, extraction slots, designing questions and few-shot examples, calibration and macro cells, prompt packs, cost (hosted) and errors; plain HTTP, the OpenAI SDK or [spinf-client](https://github.com/spinfinc/spinf-client) |
| [spinf-serve](skills/spinf-serve/SKILL.md) | running your own server | Install, models and GPUs, configuration and limits, pack sources, readiness, deployment and troubleshooting |
| [spinf-packs](skills/spinf-packs/SKILL.md) | authoring prompt packs | The pack file, calibration, validation, fixtures, the repository manifest, publishing and versioning |

Each skill is one Markdown file. Read it on GitHub, or fetch it as Markdown from `https://spinf.com/skills/<name>/SKILL.md`.
The docs index for agents is [spinf.com/llms.txt](https://spinf.com/llms.txt) (hosted) and
[spinf-serve/docs](https://github.com/spinfinc/spinf-serve/blob/main/docs/README.md) (both).

## Install

Use one installation method.

### Claude Code plugin

```bash
claude plugin marketplace add spinfinc/skills
claude plugin install spinf@spinfinc
```

The plugin installs the three skills; invoke one explicitly with `/spinf:spinf`, `/spinf:spinf-serve` or `/spinf:spinf-packs`.

### Other agents via skills.sh

```bash
npx skills add spinfinc/skills --skill spinf              # add --skill spinf-serve, --skill spinf-packs as needed
```

Select your agent when prompted. Installation is project-local by default; add `-g` to install globally.

### Manually

```bash
curl -fsSL --create-dirs -o .claude/skills/spinf/SKILL.md https://spinf.com/skills/spinf/SKILL.md
```

Or save the file in your agent's skills folder (for example `.agents/skills/spinf/SKILL.md`).

## Use

Hosted: your agent needs an API key in the `SPINF_API_KEY` environment variable ([create one](https://spinf.com/console/api-keys)).
Self-hosted: `SPINF_API_URL` points at your server (a token only if it was started with one). Then ask, for example:

> Use spinf to route incoming support tickets by topic and urgency, with human review for uncertain ones.

> Set up spinf-serve on this machine's GPU and load our team's prompt packs from GitHub.

> Turn these twenty calibrated questions into a prompt pack repository with fixtures and CI.

## License

[MIT](LICENSE).
