# spinf agent skills

Agent skills for building with [spinf](https://spinf.com): many questions about a piece of content in one call, a
probability for each answer you allow, and only input tokens billed.

Read [SKILL.md on GitHub](skills/spinf/SKILL.md), or fetch it as Markdown from
[spinf.com/skills/spinf/SKILL.md](https://spinf.com/skills/spinf/SKILL.md). The full docs index for agents is
[spinf.com/llms.txt](https://spinf.com/llms.txt).

## Install

Use one installation method.

### Claude Code plugin

```bash
claude plugin marketplace add spinfinc/skills
claude plugin install spinf@spinfinc
```

### Other agents via skills.sh

```bash
npx skills add spinfinc/skills --skill spinf
```

Select your agent when prompted. Installation is project-local by default; add `-g` to install globally.

### Manually

```bash
curl -fsSL --create-dirs -o .claude/skills/spinf/SKILL.md https://spinf.com/skills/spinf/SKILL.md
```

Or save the file in your agent's skills folder (for example `.agents/skills/spinf/SKILL.md`).

The [console](https://spinf.com/console) Quickstart has a prompt to copy into your agent.

## Use

Your agent needs an API key in the `SPINF_API_KEY` environment variable
([create one](https://spinf.com/console/api-keys)). Then ask it, for example:

> Use spinf to route incoming support tickets by topic and urgency, with human review for uncertain ones.

In Claude Code you can invoke the skill explicitly with `/spinf:spinf`.

| Skill | Purpose |
|---|---|
| [spinf](skills/spinf/SKILL.md) | Call the scoring API, read `p` / `floor_p` / `residual`, design questions and few-shot examples, estimate cost, handle errors |

## License

[MIT](LICENSE).
