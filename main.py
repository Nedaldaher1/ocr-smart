#!/usr/bin/env python3

"""
oCrai - PDF to Markdown Converter
"""
import os


def clear_screen():
    """Clears the console screen."""
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)


def print_logo():
    logo = r"""
    ____   ____  ____   ____  _     _                  
   / __ \ / __ \|  _ \ / __ \| |   (_) ___  ___  _ __ 
  | |  | | |  | | |_) | |  | | |   | |/ _ \/ _ \| '_ \
  | |__| | |__| |  _ <| |__| | |___| |  __/ (_) | | | |
   \____/ \____/|_| \_\\____/|_____|_|\___|\___/|_| |_|

    """
    print(logo)
    print("oCrai - PDF to Markdown Converter\n")


def show_menu():
    print("Please choose an option:")
    print("  1. Convert PDF to Markdown")
    print("  2. Configure Settings")
    print("  3. Help")
    print("  4. Exit")


def main():
    # Main loop with screen clearance
    while True:
        clear_screen()
        print_logo()
        show_menu()
        choice = input("Enter your choice [1-4]: ")
        if choice == "1":
            # TODO: implement PDF to Markdown conversion
            print("Converting PDF to Markdown... (not yet implemented)")
            input("\nPress Enter to continue...")
        elif choice == "2":
            # TODO: add settings configuration
            print("Opening settings... (not yet implemented)")
            input("\nPress Enter to continue...")
        elif choice == "3":
            # TODO: show help information
            print("Help:\n  - Choose 1 to convert a PDF file.\n  - Settings to adjust output.\n  - Exit to close.")
            input("\nPress Enter to continue...")
        elif choice == "4":
            clear_screen()
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please try again.\n")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
