# Proposal: Configuring Claude Code to Use Cygwin Bash

## Overview

This document proposes a configuration to make Claude Code use Cygwin Bash instead of Git Bash for
executing shell commands.  The goal is to leverage your existing Cygwin environment, which includes
your customized PATH, utilities, and shell configuration.

## Current Configuration

- **Claude Code default shell**: Git Bash (MSYS2-based), version 5.2.37
- **Git Bash executable**: `C:\Program Files\Git\bin\bash.exe`
- **Cygwin installation**: `C:\apps\cygwin`
- **Cygwin Bash executable**: `C:\apps\cygwin\bin\bash.exe`, version 5.2.21
- **Drive letter symlinks**: You have created symlinks in the Cygwin root (e.g., `/c` -> `/cygdrive/c`)
  that enable Git Bash-style absolute paths like `/c/path/to/file`

## Proposed Solution

Claude Code provides the `CLAUDE_CODE_GIT_BASH_PATH` environment variable specifically for
customizing the Bash executable on Windows systems.  Despite its name suggesting Git Bash, this
variable accepts any Bash-compatible executable path.

### Configuration Steps

#### Option 1: Set Environment Variable Permanently (Recommended)

Set the environment variable at the Windows user level so it persists across all terminal sessions:

**Using PowerShell (run as your user, not as administrator):**

```powershell
[System.Environment]::SetEnvironmentVariable(
    'CLAUDE_CODE_GIT_BASH_PATH',
    'C:\apps\cygwin\bin\bash.exe',
    'User'
)
```

**Using Windows Settings:**

1. Open Windows Settings > System > About > Advanced system settings
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Set Variable name: `CLAUDE_CODE_GIT_BASH_PATH`
5. Set Variable value: `C:\apps\cygwin\bin\bash.exe`
6. Click OK to save

After setting the environment variable, restart your terminal (and VS Code if using the extension)
for the change to take effect.

#### Option 2: Set Environment Variable in Claude Code Settings

Add the environment variable to your Claude Code `settings.json` file at
`C:\Users\flitt\.claude\settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "input=$(cat); printf '%s in %s' \"$(echo \"$input\" | jq -r '.model.display_name')\" \"$(echo \"$input\" | jq -r '.workspace.current_dir')\""
  },
  "model": "opus",
  "env": {
    "CLAUDE_CODE_GIT_BASH_PATH": "C:\\apps\\cygwin\\bin\\bash.exe"
  }
}
```

Note: Use double backslashes in JSON strings, or use forward slashes (`C:/apps/cygwin/bin/bash.exe`).

#### Option 3: Set Environment Variable Per Session (Testing)

For testing before committing to a permanent change, set the variable in your current session:

**In PowerShell:**

```powershell
$env:CLAUDE_CODE_GIT_BASH_PATH = "C:\apps\cygwin\bin\bash.exe"
claude
```

**In Git Bash or Cygwin:**

```bash
export CLAUDE_CODE_GIT_BASH_PATH="C:/apps/cygwin/bin/bash.exe"
claude
```

## Compatibility Considerations

### Path Format Compatibility

Your drive letter symlinks (`/c` -> `/cygdrive/c`, etc.) ensure that absolute paths written in the
Git Bash style (`/c/path/to/file`) work correctly in Cygwin.  This is critical because Claude Code
generates paths in this format, and your existing `CLAUDE.md` instructions specify this format.

**Verification test (executed successfully):**

```bash
$ /c/apps/cygwin/bin/bash.exe -c 'ls /c/franl | head -5'
ahk
ai
audio
backup
bin
```

### Working Directory Preservation

Claude Code expects the shell to maintain the current working directory.  Cygwin Bash behaves
correctly in this regard when invoked without the `-l` (login) flag, which is how Claude Code
invokes it.

**Verification test (executed successfully):**

```bash
# Run from C:\franl\git\ai-docs
$ /c/apps/cygwin/bin/bash.exe -c 'pwd'
/cygdrive/c/franl/git/ai-docs
```

Note: The output shows `/cygdrive/c/...` rather than `/c/...`, but this is cosmetic since the
symlinks make both forms equivalent.

### PATH Environment

When Claude Code invokes Cygwin Bash, the PATH is inherited from the parent process.  Cygwin Bash
then prepends `/usr/local/bin:/usr/bin` to this PATH (as defined in `/etc/profile`), giving Cygwin
utilities priority over Windows utilities of the same name.

Inherited PATH entries use the `/cygdrive/c/...` format after conversion.  Your personal PATH
entries (from `.bash_profile` and `.bashrc`) should work correctly.

### Potential Issues

1. **Cygwin-specific utilities**: Some Cygwin utilities may behave differently from their Git Bash
   (MSYS2) counterparts.  For example:
   - `cygpath` (Cygwin) vs. no equivalent (Git Bash uses different path handling)
   - Symlink handling may differ slightly

2. **Script compatibility**: Scripts written for Git Bash may need minor adjustments for Cygwin,
   particularly regarding:
   - Case sensitivity (Cygwin can be configured either way)
   - Line ending handling
   - Windows path conversion

3. **Login shell behavior**: If Claude Code were to invoke Bash with the `-l` flag, Cygwin would
   change to the HOME directory.  Current testing shows Claude Code does not use `-l`, so this
   should not be an issue.

## Verification Steps

After applying the configuration, verify it works correctly:

1. **Start Claude Code** in a new terminal session

2. **Ask Claude to run a test command**:
   ```
   Run: bash --version
   ```
   Expected output should show:
   ```
   GNU bash, version 5.2.21(1)-release (x86_64-pc-cygwin)
   ```

3. **Verify PATH includes Cygwin utilities**:
   ```
   Run: which cygpath
   ```
   Expected output: `/usr/bin/cygpath`

4. **Verify working directory is maintained**:
   ```
   Run: pwd
   ```
   Expected output should be the current project directory

## Rollback Procedure

If you encounter issues and need to revert to Git Bash:

**Remove the environment variable (PowerShell):**

```powershell
[System.Environment]::SetEnvironmentVariable('CLAUDE_CODE_GIT_BASH_PATH', $null, 'User')
```

**Or remove the `env` section** from `settings.json` if you used Option 2.

## Recommendation

I recommend **Option 1** (permanent environment variable) for the following reasons:

1. **Consistency**: The same Bash is used regardless of how you launch Claude Code (terminal, VS
   Code extension, etc.)

2. **Simplicity**: No need to modify Claude Code configuration files

3. **Reversibility**: Easy to remove or modify if needed

4. **Testing path**: You can first use Option 3 to test temporarily, then commit to Option 1 once
   you've verified everything works correctly

## Summary

| Configuration Method | Scope | Persistence | Recommended For |
|---------------------|-------|-------------|-----------------|
| Environment variable (User) | All sessions | Permanent | Production use |
| settings.json `env` | Claude Code only | Permanent | Claude Code-specific config |
| Export in session | Current session | Temporary | Testing |

The configuration is straightforward and leverages Claude Code's built-in support for custom Bash
executables.  Your existing drive letter symlinks ensure path compatibility between the Git Bash
path format that Claude Code uses and Cygwin's native `/cygdrive/` format.
