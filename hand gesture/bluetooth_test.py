import serial
import time

bluetooth = serial.Serial("COM5", 9600, timeout=1)

time.sleep(2)

print("Connected to Bluetooth!")

while True:
    command = input("Enter a character: ")

    if command == "q":
        break

    bluetooth.write(command.encode())

bluetooth.close()