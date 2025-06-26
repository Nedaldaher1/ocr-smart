#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
oCrai - PDF to Markdown Converter
An integrated command-line interface for converting PDF files to Markdown.
This version uses the 'rich' library for an enhanced UI and JSON for persistent settings.
"""

# --- System and Library Imports ---
import os
import sys
import shutil
import json

# --- Rich Library for beautiful CLI ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
except ImportError:
    print("Error: The 'rich' library is not installed.")
    print("Please install it by running: pip install rich")
    sys.exit(1)


# --- Import our custom processing module ---
try:
    import ocr_smarter as ocr_engine
except ImportError:
    print("Fatal Error: Could not find the module 'ocr_smarter.py'.")
    print("Please ensure both 'main.py' and 'ocr_smarter.py' are in the same directory.")
    sys.exit(1)

# --- Optional Library Imports ---
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: 'python-dotenv' is not installed. API key will not be loaded from .env file.")
    print("You can install it via: pip install python-dotenv")

# --- Global and Default Settings ---
console = Console()
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "pdf_dir": "pdfs",
    "output_dir": "output",
    "dpi": 300,
    "save_processed_scans": False,
    "clear_output": True,
}
SETTINGS = DEFAULT_SETTINGS.copy()
API_KEY_ENV_NAME = "OPENROUTER_API_KEY"
API_KEY = None


# --- Settings Management Functions ---
def save_settings():
    """Saves the current SETTINGS dictionary to the JSON file."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(SETTINGS, f, indent=4)

def load_settings():
    """Loads settings from the JSON file, or creates it with defaults if it doesn't exist."""
    global SETTINGS
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                loaded_settings = json.load(f)
                # Ensure all default keys are present
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in loaded_settings:
                        loaded_settings[key] = value
                SETTINGS = loaded_settings
            except json.JSONDecodeError:
                console.print(f"[bold red]Error: Could not decode {SETTINGS_FILE}. Using default settings.[/bold red]")
                SETTINGS = DEFAULT_SETTINGS.copy()
                save_settings() # Save a clean default file
    else:
        # If the file doesn't exist, create it with default settings
        save_settings()

# --- Command-Line Interface (CLI) UI Functions ---
def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_logo():
    """Prints the program's logo using rich."""
    logo_text = r"""
    ____   ____  ____   ____  _     _
   / __ \ / __ \|  _ \ / __ \| |   (_) ___   ___  _ __
  | |  | | |  | | |_) | |  | | |   | |/ _ \ / _ \| '_ \
  | |__| | |__| |  _ <| |__| | |___| |  __/ (_) | | | |
   \____/ \____/|_| \_\\____/|_____|_|\___|\___/|_| |_|
    """
    console.print(Align.center(f"[bold cyan]{logo_text}[/bold cyan]"))
    console.print(Align.center("[bold]oCrai - PDF to Markdown Converter[/bold]\n"))

def show_main_menu():
    """Displays the main menu using rich Panel."""
    menu_text = Text("\n")
    menu_text.append("  [1] Start PDF Conversion\n", style="bold green")
    menu_text.append("  [2] Configure Settings\n", style="yellow")
    menu_text.append("  [3] Help\n", style="cyan")
    menu_text.append("  [4] Exit\n", style="red")
    
    console.print(Panel(menu_text, title="Main Menu", border_style="blue", expand=False))
    
def show_settings_menu():
    """Displays the settings menu using rich."""
    clear_screen()
    print_logo()
    
    on_off_scans = "[green]ON[/green]" if SETTINGS['save_processed_scans'] else "[red]OFF[/red]"
    on_off_clear = "[green]ON[/green]" if SETTINGS['clear_output'] else "[red]OFF[/red]"

    settings_text = (
        f"  [1] PDF Input Directory       : [cyan]{SETTINGS['pdf_dir']}[/cyan]\n"
        f"  [2] Results Output Directory  : [cyan]{SETTINGS['output_dir']}[/cyan]\n"
        f"  [3] Scan Resolution (DPI)     : [cyan]{SETTINGS['dpi']}[/cyan]\n"
        f"  [4] Save Processed Scans      : {on_off_scans}\n"
        f"  [5] Clear Output Before Start : {on_off_clear}\n\n"
        f"  [6] Return to Main Menu"
    )
    
    console.print(Panel(settings_text, title="Settings Page", border_style="yellow"))
    console.print("\nEnter a setting number to change it, or 6 to return.")

def configure_settings():
    """Allows the user to modify and save settings."""
    while True:
        show_settings_menu()
        choice = console.input("[bold yellow]Select a setting to modify [1-6]: [/bold yellow]")

        if choice == '1':
            new_val = console.input(f"Enter new path for PDF directory (current: [cyan]{SETTINGS['pdf_dir']}[/cyan]): ")
            SETTINGS['pdf_dir'] = new_val.strip()
        elif choice == '2':
            new_val = console.input(f"Enter new path for output directory (current: [cyan]{SETTINGS['output_dir']}[/cyan]): ")
            SETTINGS['output_dir'] = new_val.strip()
        elif choice == '3':
            try:
                new_val = int(console.input(f"Enter new DPI value (current: [cyan]{SETTINGS['dpi']}[/cyan]): "))
                SETTINGS['dpi'] = new_val
            except ValueError:
                console.print("[bold red]Error: Please enter a valid integer.[/bold red]")
                console.input("Press Enter to continue...")
        elif choice == '4':
            SETTINGS['save_processed_scans'] = not SETTINGS['save_processed_scans']
        elif choice == '5':
            SETTINGS['clear_output'] = not SETTINGS['clear_output']
        elif choice == '6':
            break
        else:
            console.print("[bold red]Invalid option. Please try again.[/bold red]")
            console.input("Press Enter to continue...")
        
        save_settings() # Save after every change

def show_help():
    """Displays help information in a panel."""
    clear_screen()
    print_logo()
    help_text = (
        "This program converts educational PDF files into Markdown files using AI.\n\n"
        "[bold]Usage Steps:[/bold]\n"
        "  1. Ensure you have a `.env` file containing your API key.\n"
        "  2. Place PDF files in the directory specified in Settings (default: 'pdfs').\n"
        "  3. From the main menu, select 'Start PDF Conversion'.\n"
        "  4. A folder for each PDF will be created in the output directory (default: 'output').\n\n"
        "You can change directories and other options from the 'Configure Settings' menu. "
        "Your settings are saved in `settings.json`."
    )
    console.print(Panel(help_text, title="Help", border_style="cyan"))
    console.input("\nPress Enter to return to the main menu...")

def start_conversion():
    """Starts the full conversion process."""
    clear_screen()
    print_logo()
    if not API_KEY:
        console.print(Panel(f"[bold red]Fatal Error: API key not found.[/bold red]\nPlease ensure it exists in the .env file under the name '{API_KEY_ENV_NAME}'.", title="Error", border_style="red"))
        console.input("\nPress Enter to return to the main menu...")
        return
    
    console.print(Panel("[bold green]--- Starting Conversion Process ---[/bold green]", expand=False))
    
    # Pass the rich console object to the engine for consistent logging
    ocr_engine.run_conversion_pipeline(SETTINGS, API_KEY, console)
    
    console.input("\nPress Enter to return to the main menu...")

def main():
    """The main function to run the program."""
    global API_KEY
    load_settings() # Load settings from settings.json

    try:
        if load_dotenv():
            console.print("[green]Loaded environment variables from .env file.[/green]")
            API_KEY = os.getenv(API_KEY_ENV_NAME)
    except NameError:
        pass # python-dotenv is not installed

    while True:
        clear_screen()
        print_logo()
        show_main_menu()
        choice = console.input("[bold blue]Enter your choice [1-4]: [/bold blue]")
        
        if choice == "1":
            start_conversion()
        elif choice == "2":
            configure_settings()
        elif choice == "3":
            show_help()
        elif choice == "4":
            clear_screen()
            console.print("[bold]Goodbye![/bold]")
            break
        else:
            console.print("[bold red]Invalid selection. Please try again.[/bold red]\n")
            console.input("Press Enter to continue...")

if __name__ == "__main__":
    # Ensure default directories exist on first run
    os.makedirs(SETTINGS["pdf_dir"], exist_ok=True)
    os.makedirs(SETTINGS["output_dir"], exist_ok=True)
    main()
