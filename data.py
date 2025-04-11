import requests
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker for generating fake user data
fake = Faker()

# Base URL of your Flask app
BASE_URL = "http://localhost:5000"

# Function to register a new user
def register_user(session, fullname, phone, email, password, signup_datetime):
    url = f"{BASE_URL}/signup"
    data = {
        "fullname": fullname,
        "phone": phone,
        "email": email,
        "password": password,
        "confirm_password": password,
        "datetime": signup_datetime
    }
    response = session.post(url, data=data)
    return response

# Function to login a user
def login_user(session, email, password):
    url = f"{BASE_URL}/login"
    data = {
        "email": email,
        "password": password
    }
    response = session.post(url, data=data)
    return response

# Function to book an appointment
def book_appointment(session, fullname, appointment_date, appointment_time, reason):
    url = f"{BASE_URL}/book_appointment"
    headers = {'Content-Type': 'application/json'}
    data = {
        "fullname": fullname,
        "appointment_date": appointment_date,
        "appointment_time": appointment_time,
        "reason": reason
    }
    response = session.post(url, json=data, headers=headers)
    return response

# Create a session object to maintain cookies
session = requests.Session()

# Create 40 new users
now = datetime.now()
start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
days_diff = (now - start_of_month).days
random_days = random.randint(0, days_diff)
random_time = timedelta(
    hours=random.randint(0, 23),
    minutes=random.randint(0, 59),
    seconds=random.randint(0, 59)
)

users = []
for _ in range(40):
    fullname = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    password = fake.password()
    Date = (start_of_month + timedelta(days=random_days) + random_time).strftime("%Y-%m-%d %H:%M:%S")

    # Register the user
    response = register_user(session, fullname, phone, email, password, Date)
    if response.status_code == 200:
        users.append({"email": email, "password": password, "fullname": fullname})
    else:
        print(f"Failed to register user {fullname}: {response.text}")

# Create appointments for 30 users
appointment_reasons = ["Check-up", "Consultation", "Follow-up", "Vaccination", "Test"]
for user in users[:30]:
    # Login the user
    login_response = login_user(session, user["email"], user["password"])
    if login_response.status_code == 200:
        # Book multiple appointments
        for _ in range(random.randint(1, 5)):  # Random number of appointments between 1 and 5
            appointment_date = (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
            appointment_time = f"{random.randint(9, 16)}:{random.choice(['00', '30'])}"
            reason = random.choice(appointment_reasons)

            appointment_response = book_appointment(session, user["fullname"], appointment_date, appointment_time, reason)
            if appointment_response.status_code == 201:
                print(f"Appointment booked successfully for {user['fullname']} on {appointment_date} at {appointment_time}")
            else:
                print(f"Failed to book appointment for {user['fullname']}: {appointment_response.text}")
    else:
        print(f"Failed to login user {user['fullname']}: {login_response.text}")
