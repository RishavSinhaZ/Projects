from height_detector import HeightDetector
from gate_controller import GateController
from logger import Logger
from alert_system import AlertSystem

PARENT_PIN = "1234"  # override PIN

def main():
    gate = GateController()
    logger = Logger()
    alert = AlertSystem()

    while True:
        try:
            height = int(input("\nEnter your height in cm (or 0 to exit): "))
            if height == 0:
                break

            status = gate.check_and_open(height)
            logger.log_attempt(height, status)

            if status == "OPEN":
                print(f"✅ Gate opened for height {height} cm (Adult detected).")
            else:
                print(f"❌ Gate blocked for height {height} cm (Child detected).")
                alert.trigger_alert(height)

                # Parent override
                choice = input("Parent present? Enter PIN or press Enter to skip: ")
                if choice == PARENT_PIN:
                    print("🔑 Parent override successful. Gate opened.")
                    logger.log_attempt(height, "OVERRIDE OPEN")
                elif choice != "":
                    print("❌ Wrong PIN. Gate remains closed.")
                    logger.log_attempt(height, "OVERRIDE FAILED")

        except ValueError:
            print("⚠️ Please enter a valid number for height.")

    print("\n--- LOG HISTORY ---")
    logger.show_logs()

if __name__ == "__main__":
    main()