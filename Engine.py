#!/usr/bin/env python3
“””
Enterprise Console Application with GitHub Authentication
Features: Autocomplete, History, Multiple Output Formats, HTTPS, GitHub Auth
“””

import os
import sys
import json
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Core libraries

import readline
import atexit
from urllib.parse import urlencode, parse_qs

# Rich console output

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress
from rich.syntax import Syntax

# Web server for OAuth callback

from aiohttp import web, ClientSession
import ssl
import certifi

# Command parsing

import argparse
import shlex

# Data formatting

import pandas as pd
import tabulate

@dataclass
class GitHubUser:
“”“GitHub user information”””
login: str
name: str
email: str
avatar_url: str
company: Optional[str] = None

@dataclass
class AppConfig:
“”“Application configuration”””
github_client_id: str
github_client_secret: str
github_enterprise_url: Optional[str] = None
history_file: str = “.console_history”
log_file: str = “app.log”
ssl_cert: Optional[str] = None
ssl_key: Optional[str] = None
port: int = 8443

class GitHubAuth:
“”“GitHub Enterprise authentication handler”””

```
def __init__(self, config: AppConfig):
    self.config = config
    self.access_token: Optional[str] = None
    self.user: Optional[GitHubUser] = None
    self.base_url = config.github_enterprise_url or "https://github.com"
    self.api_url = f"{self.base_url}/api/v3" if config.github_enterprise_url else "https://api.github.com"

async def authenticate(self) -> bool:
    """Perform OAuth authentication flow"""
    console = Console()
    
    # Generate OAuth URL
    oauth_url = f"{self.base_url}/login/oauth/authorize"
    params = {
        "client_id": self.config.github_client_id,
        "scope": "user:email",
        "state": "secure_random_state"
    }
    
    auth_url = f"{oauth_url}?{urlencode(params)}"
    
    console.print(f"[bold blue]Please visit this URL to authenticate:[/bold blue]")
    console.print(f"[link]{auth_url}[/link]")
    
    # Start callback server
    callback_received = asyncio.Event()
    auth_code = None
    
    async def handle_callback(request):
        nonlocal auth_code
        query = parse_qs(request.query_string)
        if 'code' in query:
            auth_code = query['code'][0]
            callback_received.set()
            return web.Response(text="Authentication successful! You can close this window.")
        return web.Response(text="Authentication failed!", status=400)
    
    app = web.Application()
    app.router.add_get('/callback', handle_callback)
    
    # Setup SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    if self.config.ssl_cert and self.config.ssl_key:
        ssl_context.load_cert_chain(self.config.ssl_cert, self.config.ssl_key)
    else:
        # Use self-signed certificate for development
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', self.config.port, ssl_context=ssl_context)
    await site.start()
    
    console.print(f"[yellow]Waiting for authentication callback on https://localhost:{self.config.port}/callback[/yellow]")
    console.print("[dim]Note: You may see a security warning due to self-signed certificate[/dim]")
    
    try:
        await asyncio.wait_for(callback_received.wait(), timeout=300)  # 5 minutes timeout
    except asyncio.TimeoutError:
        console.print("[red]Authentication timeout[/red]")
        return False
    finally:
        await runner.cleanup()
    
    if not auth_code:
        return False
    
    # Exchange code for access token
    return await self._exchange_code_for_token(auth_code)

async def _exchange_code_for_token(self, code: str) -> bool:
    """Exchange authorization code for access token"""
    token_url = f"{self.base_url}/login/oauth/access_token"
    
    data = {
        "client_id": self.config.github_client_id,
        "client_secret": self.config.github_client_secret,
        "code": code
    }
    
    async with ClientSession() as session:
        async with session.post(token_url, data=data, headers={"Accept": "application/json"}) as response:
            if response.status == 200:
                result = await response.json()
                self.access_token = result.get("access_token")
                return await self._get_user_info()
    
    return False

async def _get_user_info(self) -> bool:
    """Get user information from GitHub API"""
    if not self.access_token:
        return False
    
    headers = {
        "Authorization": f"token {self.access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    async with ClientSession() as session:
        async with session.get(f"{self.api_url}/user", headers=headers) as response:
            if response.status == 200:
                user_data = await response.json()
                self.user = GitHubUser(
                    login=user_data["login"],
                    name=user_data.get("name", ""),
                    email=user_data.get("email", ""),
                    avatar_url=user_data.get("avatar_url", ""),
                    company=user_data.get("company")
                )
                return True
    
    return False
```

class CommandHistory:
“”“Command history management”””

```
def __init__(self, history_file: str):
    self.history_file = Path(history_file)
    self.load_history()
    atexit.register(self.save_history)

def load_history(self):
    """Load command history from file"""
    if self.history_file.exists():
        readline.read_history_file(str(self.history_file))

def save_history(self):
    """Save command history to file"""
    readline.write_history_file(str(self.history_file))

def add_command(self, command: str):
    """Add command to history"""
    readline.add_history(command)
```

class OutputFormatter:
“”“Format output in various formats”””

```
def __init__(self, console: Console):
    self.console = console

def format_text(self, data: Any, title: str = None) -> None:
    """Format as plain text"""
    if title:
        self.console.print(f"[bold]{title}[/bold]")
    self.console.print(str(data))

def format_table(self, data: List[Dict], title: str = None) -> None:
    """Format as rich table"""
    if not data:
        self.console.print("[yellow]No data to display[/yellow]")
        return
    
    table = Table(title=title)
    
    # Add columns
    for key in data[0].keys():
        table.add_column(str(key).title())
    
    # Add rows
    for row in data:
        table.add_row(*[str(v) for v in row.values()])
    
    self.console.print(table)

def format_json(self, data: Any, title: str = None) -> None:
    """Format as JSON"""
    if title:
        self.console.print(f"[bold]{title}[/bold]")
    
    json_str = json.dumps(data, indent=2, default=str)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    self.console.print(syntax)

def format_dataframe(self, df: pd.DataFrame, title: str = None) -> None:
    """Format pandas DataFrame"""
    if title:
        self.console.print(f"[bold]{title}[/bold]")
    
    # Convert to tabulate format
    table_str = tabulate.tabulate(df, headers='keys', tablefmt='grid', showindex=False)
    self.console.print(table_str)
```

class CommandCompleter:
“”“Command autocompletion”””

```
def __init__(self, commands: Dict[str, Any]):
    self.commands = commands
    self.setup_completion()

def setup_completion(self):
    """Setup readline completion"""
    readline.set_completer(self.complete)
    readline.parse_and_bind('tab: complete')

def complete(self, text: str, state: int) -> Optional[str]:
    """Completion function"""
    if state == 0:
        # First call - generate completions
        line = readline.get_line_buffer()
        words = line.split()
        
        if not words or (len(words) == 1 and not line.endswith(' ')):
            # Complete command names
            self.matches = [cmd for cmd in self.commands.keys() if cmd.startswith(text)]
        else:
            # Complete subcommands or parameters
            cmd = words[0]
            if cmd in self.commands and hasattr(self.commands[cmd], 'subcommands'):
                self.matches = [sub for sub in self.commands[cmd].subcommands if sub.startswith(text)]
            else:
                self.matches = []
    
    try:
        return self.matches[state]
    except IndexError:
        return None
```

class ConsoleApp:
“”“Main console application”””

```
def __init__(self, config: AppConfig):
    self.config = config
    self.console = Console()
    self.auth = GitHubAuth(config)
    self.history = CommandHistory(config.history_file)
    self.formatter = OutputFormatter(self.console)
    self.authenticated = False
    
    # Setup logging
    logging.basicConfig(
        filename=config.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    self.logger = logging.getLogger(__name__)
    
    # Define commands
    self.commands = {
        'help': self.cmd_help,
        'exit': self.cmd_exit,
        'quit': self.cmd_exit,
        'status': self.cmd_status,
        'user': self.cmd_user,
        'history': self.cmd_history,
        'clear': self.cmd_clear,
        'format': self.cmd_format,
        'sample': self.cmd_sample,
        'test': self.cmd_test,
    }
    
    # Setup completion
    self.completer = CommandCompleter(self.commands)

async def start(self):
    """Start the application"""
    self.console.print(Panel.fit(
        "[bold blue]Enterprise Console Application[/bold blue]\n"
        "Type 'help' for available commands",
        title="Welcome"
    ))
    
    # Authenticate
    self.console.print("\n[yellow]Authenticating with GitHub...[/yellow]")
    if await self.auth.authenticate():
        self.authenticated = True
        self.console.print(f"[green]✓ Authenticated as {self.auth.user.login}[/green]")
        self.logger.info(f"User {self.auth.user.login} authenticated successfully")
    else:
        self.console.print("[red]✗ Authentication failed[/red]")
        if not Confirm.ask("Continue without authentication?"):
            return
    
    # Main loop
    await self.main_loop()

async def main_loop(self):
    """Main command loop"""
    while True:
        try:
            # Get user input with prompt
            prompt_text = f"[{self.auth.user.login if self.auth.user else 'guest'}]> "
            command = Prompt.ask(prompt_text, default="").strip()
            
            if not command:
                continue
            
            # Add to history
            self.history.add_command(command)
            
            # Parse and execute command
            await self.execute_command(command)
            
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            self.logger.error(f"Command execution error: {e}")

async def execute_command(self, command_line: str):
    """Execute a command"""
    try:
        parts = shlex.split(command_line)
    except ValueError:
        self.console.print("[red]Invalid command syntax[/red]")
        return
    
    if not parts:
        return
    
    command = parts[0].lower()
    args = parts[1:]
    
    if command in self.commands:
        await self.commands[command](args)
    else:
        self.console.print(f"[red]Unknown command: {command}[/red]")
        self.console.print("Type 'help' for available commands")

async def cmd_help(self, args: List[str]):
    """Show help information"""
    help_table = Table(title="Available Commands")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")
    
    commands_help = {
        "help": "Show this help message",
        "exit/quit": "Exit the application",
        "status": "Show application status",
        "user": "Show user information",
        "history": "Show command history",
        "clear": "Clear the screen",
        "format": "Demonstrate output formats",
        "sample": "Show sample data in different formats",
        "test": "Run connectivity tests"
    }
    
    for cmd, desc in commands_help.items():
        help_table.add_row(cmd, desc)
    
    self.console.print(help_table)

async def cmd_exit(self, args: List[str]):
    """Exit the application"""
    self.console.print("[yellow]Goodbye![/yellow]")
    sys.exit(0)

async def cmd_status(self, args: List[str]):
    """Show application status"""
    status_data = {
        "Authenticated": "✓ Yes" if self.authenticated else "✗ No",
        "User": self.auth.user.login if self.auth.user else "Guest",
        "GitHub URL": self.auth.base_url,
        "History File": self.config.history_file,
        "Log File": self.config.log_file,
        "Port": str(self.config.port),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    status_table = Table(title="Application Status")
    status_table.add_column("Property", style="cyan")
    status_table.add_column("Value", style="white")
    
    for key, value in status_data.items():
        status_table.add_row(key, value)
    
    self.console.print(status_table)

async def cmd_user(self, args: List[str]):
    """Show user information"""
    if not self.auth.user:
        self.console.print("[yellow]Not authenticated[/yellow]")
        return
    
    user_data = {
        "Login": self.auth.user.login,
        "Name": self.auth.user.name or "N/A",
        "Email": self.auth.user.email or "N/A",
        "Company": self.auth.user.company or "N/A",
        "Avatar": self.auth.user.avatar_url
    }
    
    self.formatter.format_table([user_data], "User Information")

async def cmd_history(self, args: List[str]):
    """Show command history"""
    history_length = readline.get_current_history_length()
    history_items = []
    
    for i in range(max(0, history_length - 10), history_length):
        item = readline.get_history_item(i + 1)
        if item:
            history_items.append({
                "Index": i + 1,
                "Command": item,
                "Time": "Recent"
            })
    
    if history_items:
        self.formatter.format_table(history_items, "Command History (Last 10)")
    else:
        self.console.print("[yellow]No command history available[/yellow]")

async def cmd_clear(self, args: List[str]):
    """Clear the screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

async def cmd_format(self, args: List[str]):
    """Demonstrate different output formats"""
    sample_data = [
        {"Name": "John Doe", "Age": 30, "City": "New York"},
        {"Name": "Jane Smith", "Age": 25, "City": "London"},
        {"Name": "Bob Johnson", "Age": 35, "City": "Tokyo"}
    ]
    
    format_type = args[0] if args else "table"
    
    if format_type == "table":
        self.formatter.format_table(sample_data, "Sample Data - Table Format")
    elif format_type == "json":
        self.formatter.format_json(sample_data, "Sample Data - JSON Format")
    elif format_type == "text":
        self.formatter.format_text(sample_data, "Sample Data - Text Format")
    elif format_type == "dataframe":
        df = pd.DataFrame(sample_data)
        self.formatter.format_dataframe(df, "Sample Data - DataFrame Format")
    else:
        self.console.print("[red]Invalid format. Use: table, json, text, or dataframe[/red]")

async def cmd_sample(self, args: List[str]):
    """Show sample data"""
    self.console.print("[bold]Demonstrating all output formats:[/bold]\n")
    
    # Show in all formats
    for fmt in ["table", "json", "text", "dataframe"]:
        await self.cmd_format([fmt])
        self.console.print()

async def cmd_test(self, args: List[str]):
    """Run connectivity tests"""
    self.console.print("[bold]Running connectivity tests...[/bold]")
    
    with Progress() as progress:
        task = progress.add_task("Testing connections...", total=3)
        
        # Test GitHub API
        try:
            async with ClientSession() as session:
                async with session.get(f"{self.auth.api_url}/user", 
                                     headers={"Authorization": f"token {self.auth.access_token}"} if self.auth.access_token else {}) as response:
                    github_status = "✓ Connected" if response.status == 200 else f"✗ Error ({response.status})"
        except Exception as e:
            github_status = f"✗ Error: {e}"
        
        progress.update(task, advance=1)
        
        # Test HTTPS
        https_status = "✓ Available" if self.config.ssl_cert else "⚠ Self-signed only"
        progress.update(task, advance=1)
        
        # Test file access
        try:
            with open(self.config.log_file, 'a') as f:
                f.write(f"Test entry - {datetime.now()}\n")
            file_status = "✓ Writable"
        except Exception as e:
            file_status = f"✗ Error: {e}"
        
        progress.update(task, advance=1)
    
    # Show results
    test_results = [
        {"Test": "GitHub API", "Status": github_status},
        {"Test": "HTTPS Support", "Status": https_status},
        {"Test": "File Access", "Status": file_status}
    ]
    
    self.formatter.format_table(test_results, "Connectivity Test Results")
```

def load_config() -> AppConfig:
“”“Load configuration from environment variables”””
return AppConfig(
github_client_id=os.getenv(“GITHUB_CLIENT_ID”, “”),
github_client_secret=os.getenv(“GITHUB_CLIENT_SECRET”, “”),
github_enterprise_url=os.getenv(“GITHUB_ENTERPRISE_URL”),
history_file=os.getenv(“HISTORY_FILE”, “.console_history”),
log_file=os.getenv(“LOG_FILE”, “app.log”),
ssl_cert=os.getenv(“SSL_CERT_FILE”),
ssl_key=os.getenv(“SSL_KEY_FILE”),
port=int(os.getenv(“PORT”, “8443”))
)

async def main():
“”“Main entry point”””
config = load_config()

```
if not config.github_client_id or not config.github_client_secret:
    print("Error: GitHub client ID and secret must be provided via environment variables")
    print("Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET")
    sys.exit(1)

app = ConsoleApp(config)
await app.start()
```

if **name** == “**main**”:
try:
asyncio.run(main())
except KeyboardInterrupt:
print(”\nApplication interrupted by user”)
sys.exit(0)
