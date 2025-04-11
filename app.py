# app.py
from flask import Flask, render_template, request, jsonify
from flask_pymongo import PyMongo
from config import Config
from flask import Flask, render_template, request, redirect, flash
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from bson.objectid import ObjectId
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from appointment_model import AppointmentModel
import logging
from datetime import datetime
from bot import MedicalChatbot
from bson.objectid import ObjectId

app = Flask(__name__)
app.config.from_object(Config)

mongo = PyMongo(app)
users = mongo.db.users
medical_chatbot = MedicalChatbot()
medical_chatbot.mongo = mongo

# current_dir = os.path.dirname(__file__)
# project_root = os.path.dirname(os.path.dirname(current_dir))
# data_dir = os.path.join(project_root, 'Practice_Level_Crosstab_Jan_24')
data_dir = Config.DATA_DIR
appointment_model = AppointmentModel(data_dir)

# with h5py.File('best_model.h5', 'r') as f:
#     print(f.keys())

# # Load model, tokenizer, label encoder
# model = load_model('best_model.h5')

# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Basic validations
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect('/signup')
            
        if users.find_one({"email": email}):
            flash("Email already registered!", "warning")
            return redirect('/signup')

        # Hash the password before storing
        hashed_password = generate_password_hash(password)

        # Insert into MongoDB
        users.insert_one({
            "fullname": fullname,
            "phone": phone,
            "email": email,
            "password": hashed_password,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        flash("Signup successful! Please login.", "success")
        return redirect('/login')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = mongo.db.users.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            session['user'] = {
                '_id': str(user['_id']),
                'username': user['fullname'],
                'email': user['email']
            }
            session['username'] = user['fullname']
            session['_id'] = str(user['_id'])
            session['email'] = str(user['email'])
            flash(f"Welcome back, {user['fullname']}!", "success")
            return redirect('/dashboard')  
        else:
            flash("Invalid email or password", "danger")
            return redirect('/login')

    return render_template('login.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        admin = mongo.db.admins.find_one({'email': email})

        if admin and check_password_hash(admin['password'], password):
            session['admin'] = admin['fullname']
            flash(f"Welcome Admin, {admin['fullname']}!", "success")
            return redirect('/admin-dashboard')
        else:
            flash("Invalid admin credentials", "danger")
            return redirect('/admin')

    return render_template('admin_login.html')


@app.route('/admin-dashboard')
def admin_dashboard():
    if 'admin' not in session:
        flash("Unauthorized Access!", "danger")
        return redirect('/admin')
    return render_template('admin_dashboard.html', admin=session['admin'])

@app.route('/register-admin', methods=['POST'])
def register_admin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    fullname = data.get('fullname')

    if not email or not password or not fullname:
        return jsonify({"error": "Missing fields"}), 400

    if mongo.db.admins.find_one({"email": email}):
        return jsonify({"error": "Admin already exists"}), 409

    hashed_pw = generate_password_hash(password)
    mongo.db.admins.insert_one({
        "email": email,
        "password": hashed_pw,
        "fullname": fullname
    })

    return jsonify({"message": "Admin registered successfully"}), 201

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['new_password']

        user = mongo.db.users.find_one({'email': email})
        if user:
            hashed_pw = generate_password_hash(new_password)
            mongo.db.users.update_one({'email': email}, {'$set': {'password': hashed_pw}})
            flash("Password updated successfully!", "success")
            return redirect('/login')
        else:
            flash("Email not found!", "danger")
            return redirect('/forgot-password')

    return render_template('forgot_password.html', is_admin=False)

@app.route('/admin-forgot-password', methods=['GET', 'POST'])
def admin_forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['new_password']

        admin = mongo.db.admins.find_one({'email': email})
        if admin:
            hashed_pw = generate_password_hash(new_password)
            mongo.db.admins.update_one({'email': email}, {'$set': {'password': hashed_pw}})
            flash("Admin password updated successfully!", "success")
            return redirect('/admin')
        else:
            flash("Admin email not found!", "danger")
            return redirect('/admin-forgot-password')

    return render_template('forgot_password.html', is_admin=True)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect('/login')
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect('/')

@app.route('/profile')
def profile():
    if 'user' not in session:
        flash("Please log in first.", "warning")
        return redirect('/login')

    user_data = mongo.db.users.find_one({'fullname': session['username']})
    return render_template('profile.html', user=user_data)

@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user' not in session:
        flash("Please log in to edit profile.", "warning")
        return redirect('/login')

    user = mongo.db.users.find_one({'fullname': session['username']})

    if request.method == 'POST':
        updated_fullname = request.form['fullname']
        updated_phone = request.form['phone']
        updated_email = request.form['email']

        # Update the user record
        mongo.db.users.update_one(
            {'_id': user['_id']},
            {'$set': {
                'fullname': updated_fullname,
                'phone': updated_phone,
                'email': updated_email
            }}
        )

        # Update session name if fullname is changed
        session['username'] = updated_fullname

        flash("Profile updated successfully!", "success")
        return redirect('/profile')

    return render_template('edit_profile.html', user=user)

@app.route('/my-appointments')
def my_appointments():
    if 'user' not in session:
        flash("Please log in to view your appointments.", "warning")
        return redirect('/login')

    user_id = session.get('_id')
    # if not user_id:
    #     flash("User ID not found. Please log in again.", "warning")
    #     return redirect('/login')

    # Fetch appointments for the logged-in user
    appointments = list(mongo.db.appointments.find({'user_id': user_id}))

    return render_template('my_appointments.html', appointments=appointments)

@app.route('/admin/users')
def view_users():
    if 'admin' not in session:
        flash("Unauthorized Access!", "danger")
        return redirect('/admin')

    users_list = mongo.db.users.find()
    return render_template('admin_users.html', users=users_list)

# @app.route('/admin/appointments')
# def view_appointments():
#     if 'admin' not in session:
#         flash("Unauthorized Access!", "danger")
#         return redirect('/admin')

#     appointments = mongo.db.appointments.find()
#     return render_template('admin_appointments.html', appointments=appointments)

@app.route('/admin/appointments')
def admin_appointments():
    if 'admin' not in session:
        flash("Unauthorized Access!", "danger")
        return redirect('/admin')
    appointments = list(mongo.db.appointments.find())
    return render_template('admin_appointments.html', appointments=appointments)

@app.route('/admin/appointments/cancel/<appointment_id>', methods=['POST'])
def cancel_appointment(appointment_id):
    if 'admin' not in session:
        flash("Unauthorized Access!", "danger")
        return redirect('/admin')
    mongo.db.appointments.delete_one({'_id': ObjectId(appointment_id)})
    flash("Appointment cancelled successfully!", "success")
    return redirect('/admin/appointments')

@app.route('/chat')
def chatbot():
    if 'user' not in session:
        flash("Please log in to access the chatbot.", "warning")
        return redirect('/login')
    return render_template('chatbot.html')

@app.route('/chatbot-response', methods=['POST'])
def chatbot_response():
    if 'user' not in session:
        return jsonify({'error': 'Please login first'}), 401

    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'reply': "Please say something!"}), 400

        # Set user information for the chatbot
        user_data = {
            'user_id': str(session.get('_id')),
            'user_name': session.get('user'),
            'email': session.get('email')
        }
        medical_chatbot.set_user(user_data)

        # Generate response
        response = medical_chatbot.generate_response(user_message)

        return jsonify({
            'reply': response,
            'success': True
        })

    except Exception as e:
        app.logger.error(f"Chatbot error: {str(e)}")
        return jsonify({
            'reply': "I apologize, but I encountered an error. Please try again.",
            'success': False,
            'error': str(e)
        }), 500



@app.route('/book')
def book_page():
    if 'user' not in session:
        flash("Please login to book an appointment.", "warning")
        return redirect('/login')
    return render_template('book_appointment.html')


# @app.route('/book_appointment', methods=['POST'])
# def book_appointment():
#     data = request.json
#     # Save data to MongoDB
#     mongo.db.appointments.insert_one(data)
#     return jsonify({"message": "Appointment booked successfully!"}), 201

@app.route('/api/predict_appointments', methods=['GET'])
def predict_appointments():
    try:
        days = request.args.get('days', default=30, type=int)
        start_date = request.args.get('start_date', default=None)
        
        predictions = appointment_model.predict_appointments(
            start_date=start_date,
            days=days
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model_metrics', methods=['GET'])
def get_model_metrics():
    try:
        metrics = appointment_model.get_model_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/refresh_model', methods=['POST'])
def refresh_model():
    try:
        appointment_model.process_raw_data()
        return jsonify({
            'success': True,
            'message': 'Model refreshed successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# book_appointment route
@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    try:
        if 'user' not in session:
            return jsonify({
                "success": False,
                "error": "Please login first"
            }), 401

        data = request.json
        logger.info(f"Received booking request: {data}")

        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Validate required fields
        required_fields = ['fullname', 'appointment_date', 'appointment_time', 'reason']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Initialize the chatbot
        chatbot = MedicalChatbot()
        
        # Get user information from session
        user_id = str(session.get('_id'))
        
        # Set the current user
        user_data = {
            'user_id': user_id,
            'user_name': data['fullname']
        }
        chatbot.set_user(user_data)

        # Generate reference number first
        reference = chatbot._generate_reference()

        # Check if the slot is available
        available_slots = chatbot.check_availability(data['appointment_date'])
        if data['appointment_time'] not in available_slots:
            return jsonify({
                "success": False,
                "error": "Selected time slot is not available",
                "available_slots": available_slots
            }), 400

        # Create base appointment data
        appointment_data = {
            'reference': reference,
            'user_id': user_id,
            'user_name': data['fullname'],
            'appointment_date': data['appointment_date'],
            'appointment_time': data['appointment_time'],
            'reason': data['reason'],
            'status': 'scheduled',
            'created_at': datetime.now()
        }

        # Get prediction
        try:
            prediction = appointment_model.predict_appointments(
                start_date=data['appointment_date'],
                days=1
            )[0]
            
            appointment_data.update({
                'predicted_load': prediction['predicted_appointments'],
                'confidence_score': prediction['confidence_score']
            })
        except Exception as e:
            logger.warning(f"Prediction error: {str(e)}")
            appointment_data.update({
                'predicted_load': 15,
                'confidence_score': 0.7
            })

        # Set the appointment
        appointment_response = chatbot.set_appointment(
            date=data['appointment_date'],
            time=data['appointment_time'],
            text=data['reason'],
            user_id=user_id,
            user_name=data['fullname']
        )

        if isinstance(appointment_response, dict) and 'error' not in appointment_response:
            # Save to MongoDB
            result = mongo.db.appointments.insert_one(appointment_data)

            if result.inserted_id:
                logger.info(f"Appointment booked successfully: {result.inserted_id}")
                response_data = {
                    "success": True,
                    "message": f"Appointment booked successfully! Your reference number is {reference}",
                    "appointment_id": str(result.inserted_id),
                    "reference": reference,
                    "appointment": {
                        "date": data['appointment_date'],
                        "time": data['appointment_time'],
                        "reference": reference,
                        "fullname": data['fullname'],
                        "reason": data['reason']
                    },
                    # "prediction": {
                    #     "predicted_appointments": appointment_data['predicted_load'],
                    #     "confidence_score": appointment_data['confidence_score']
                    # }
                }
                
                return jsonify(response_data), 201

            raise Exception("Failed to save appointment")
        else:
            error_message = appointment_response.get('message') if isinstance(appointment_response, dict) else str(appointment_response)
            return jsonify({
                "success": False,
                "error": error_message
            }), 400

    except Exception as e:
        logger.error(f"Booking error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
@app.route('/api/check_availability', methods=['GET'])
def check_availability():
    try:
        date = request.args.get('date')
        if not date:
            return jsonify({
                'success': False,
                'error': 'Date parameter is required'
            }), 400

        prediction = appointment_model.predict_appointments(
            start_date=date,
            days=1
        )[0]

        availability_status = 'high'
        if prediction['predicted_appointments'] > 20:
            availability_status = 'low'
        elif prediction['predicted_appointments'] > 10:
            availability_status = 'medium'

        return jsonify({
            'success': True,
            'availability': {
                'status': availability_status,
                'predicted_appointments': prediction['predicted_appointments'],
                'confidence_score': prediction['confidence_score']
            }
        })

    except Exception as e:
        logger.error(f"Error checking availability: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Not Found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal Server Error'
    }), 50

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analytics')
def analytics():
    if 'admin' not in session:
        flash("Unauthorized Access!", "danger")
        return redirect('/admin')
    return render_template('analytics.html')

@app.route('/api/analytics/data')
def get_analytics_data():
    if 'admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Get current month and previous 11 months
        current_date = datetime.now()
        months = []
        for i in range(11, -1, -1):
            month_date = current_date.replace(day=1)
            month_date = month_date.replace(month=((current_date.month - i - 1) % 12) + 1)
            if month_date.month > current_date.month:
                month_date = month_date.replace(year=current_date.year - 1)
            months.append(month_date.strftime("%Y-%m"))

        # Get all users
        total_users = mongo.db.users.count_documents({})
        
        # Get active users (users with appointments)
        active_users_ids = set()
        appointments_data = list(mongo.db.appointments.find())
        
        for appointment in appointments_data:
            active_users_ids.add(appointment.get('user_id'))
        
        active_users_count = len(active_users_ids)
        inactive_users_count = total_users - active_users_count

        # Get registered users data
        users_data = list(mongo.db.users.find())
        registered_users = {}
        for user in users_data:
            month = ObjectId(user['_id']).generation_time.strftime("%Y-%m")
            registered_users[month] = registered_users.get(month, 0) + 1

        # Get appointments by month
        appointments_by_month = {}
        for appointment in appointments_data:
            month = ObjectId(appointment['_id']).generation_time.strftime("%Y-%m")
            appointments_by_month[month] = appointments_by_month.get(month, 0) + 1

        def fill_missing_months(data_dict):
            return [data_dict.get(month, 0) for month in months]

        return jsonify({
            'success': True,
            'data': {
                'registered_users': {
                    'labels': months,
                    'data': fill_missing_months(registered_users)
                },
                'active_users': {
                    'labels': ['Active Users', 'Inactive Users'],
                    'data': [active_users_count, inactive_users_count]
                },
                'appointments': {
                    'labels': months,
                    'data': fill_missing_months(appointments_by_month)
                }
            }
        })

    except Exception as e:
        app.logger.error(f"Analytics error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
