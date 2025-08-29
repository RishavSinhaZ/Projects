from utils import check_master_password, generate_otp
import manager

def main():
    print("🔐 Welcome to Skyler's Password Manager 🔐")
    
    if not check_master_password():
        return
    
    if not generate_otp():
        return
    
    while True:
        print("\n📌 Menu:")
        print("1. Add Password")
        print("2. View Password")
        print("3. Delete Password")
        print("4. List Services")
        print("5. Export Passwords")
        print("6. Import Passwords")
        print("8. Edit Password")
        print("9. Change Master Password")
        print("7. Exit")

        choice = input("Choose option: ")
        
        if choice == "1":
            service = input("Service: ")
            username = input("Username/Email:  ")
            password = input("Password: ")
            manager.add_password(service, username, password)
        elif choice == "2":
            service = input("Service to view: ")
            manager.view_password(service)
        elif choice == "3":
            service = input("Service to delete: ")
            manager.delete_password(service)
        elif choice == "4":
            manager.list_services()
        elif choice == "5":
            manager.export_passwords()
        elif choice == "6":
            manager.import_passwords()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        elif choice == "2":
            service = input("Service to view: ").strip().lower()
            manager.view_password(service)
        elif choice == "8":
            service = input("Service: ")
            username = input("Username to edit: ")
            new_username = input("New username (leave blank to keep same): ") or None
            new_password = input("New password (leave blank to keep same): ") or None
            manager.edit_password(service, username, new_username, new_password)

        elif choice == "9":
            old_master = input("Enter old master password: ")
            new_master = input("Enter new master password: ")
            manager.change_master_password(old_master, new_master)

        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
