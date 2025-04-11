import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import tensorflow as tf
from datetime import datetime, timedelta
import spacy
import random
import json

class MedicalChatbot:
    def __init__(self):
        # Load spaCy model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Mock database for appointments
        self.appointments_db = {}
        self.available_slots = self._generate_available_slots()
        self.current_user = None 
        
        # Enhanced intents with more variations
        self.intents = {
            'greet': [
                'hello', 'hi', 'hey', 'good morning', 'good evening', 'hi there',
                'hello doctor', 'hey there', 'greetings'
            ],
            'goodbye': [
                'bye', 'goodbye', 'see you', 'take care', 'thanks bye',
                'thank you goodbye', 'catch you later', 'have a good day'
            ],
            'book_appointment': [
                'book appointment', 'make appointment', 'schedule appointment',
                'book consultation', 'need to see doctor', 'want to book',
                'can i get an appointment', 'need a consultation'
            ],
            'cancel_appointment': [
                'cancel appointment', 'cancel booking', 'delete appointment',
                'remove booking', 'want to cancel', 'drop my appointment'
            ],
            'reschedule_appointment': [
                'reschedule appointment', 'change appointment', 'move appointment',
                'change booking time', 'want to reschedule', 'different time'
            ],
            'check_availability': [
                'check availability', 'available slots', 'when available',
                'free slots', 'doctor available', 'what times are free',
                'show me available times', 'when can i come'
            ],
            'check_prediction': [
                'how busy will it be', 'predict appointments', 'appointment forecast',
                'expected bookings', 'prediction', 'forecast'
            ]
        }

        # State management
        self.current_booking = {
            'date': None,
            'time': None,
            'in_progress': False
        }

        # Initialize vectorizer and train
        self._initialize_vectorizer()
        
        # Load prediction model
        try:
            self.prediction_model = tf.keras.models.load_model('gp_appointment_prediction_model.keras')
            self.model_loaded = True
        except:
            print("Warning: Prediction model not found. Running without prediction capabilities.")
            self.model_loaded = False

    def _initialize_vectorizer(self):
        # Prepare training data for improved vectorization
        self.training_data = []
        self.labels = []
        for intent, phrases in self.intents.items():
            self.training_data.extend(phrases)
            self.labels.extend([intent] * len(phrases))
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self.vectors = self.vectorizer.fit_transform(self.training_data)

    def _generate_available_slots(self):
        # Generate mock available slots for next 7 days
        slots = {}
        for i in range(7):
            date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            slots[date] = [
                '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
                '14:00', '14:30', '15:00', '15:30', '16:00', '16:30'
            ]
        return slots

    def _extract_entities(self, text):
        doc = self.nlp(text)
        entities = {
            'date': None,
            'time': None,
            'reference': None
        }

        # Extract date
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                try:
                    date_text = ent.text.lower()
                    if 'tomorrow' in date_text:
                        entities['date'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                    elif 'today' in date_text:
                        entities['date'] = datetime.now().strftime('%Y-%m-%d')
                    # Add more date patterns as needed
                except:
                    pass

        # Extract time using regex
        time_pattern = r'([0-9]{1,2}):([0-9]{2})'
        time_match = re.search(time_pattern, text)
        if time_match:
            entities['time'] = time_match.group()

        # Extract reference number
        ref_pattern = r'REF[0-9]{6}'
        ref_match = re.search(ref_pattern, text)
        if ref_match:
            entities['reference'] = ref_match.group()

        return entities

    def _generate_reference(self):
        return f"REF{random.randint(100000, 999999)}"
    
    def set_user(self, user_data):
        """Set current user information"""
        self.current_user = user_data

    def set_appointment(self, date, time, text, user_id, user_name):
        try:
            if date in self.available_slots and time in self.available_slots[date]:
                reference = self._generate_reference()

                # Ensure the reference ID is unique
                while reference in self.appointments_db:
                    reference = self._generate_reference()
                
                appointment_data = {
                    'reference': reference,
                    'user_id': user_id,
                    'user_name': user_name,
                    'date': date,
                    'time': time,
                    'status': 'confirmed',
                    'created_at': datetime.now(),
                    'reason': text
                }
                
                # Save to MongoDB if available
                if hasattr(self, 'mongo') and self.mongo:
                    self.mongo.db.appointments.insert_one(appointment_data)
                
                self.appointments_db[reference] = appointment_data
                self.available_slots[date].remove(time)
                self.current_booking['in_progress'] = False
                
                return {
                    'message': f"Appointment confirmed! Your reference number is {reference}. Please keep this for future reference.",
                    'appointment': appointment_data
                }
            else:
                return {
                    'message': "Sorry, that slot is not available. Would you like to see other available times?",
                    'available_slots': self.available_slots[date] if date in self.available_slots else []
                }
        except Exception as e:
            print(f"Error setting appointment: {str(e)}")
            self.current_booking['in_progress'] = False
            return {
                'message': f"Sorry, there was an error booking your appointment: {str(e)}",
                'error': True
            }
        
    def remove_appointment(self, reference, user_id, user_name):
        try:
            # Check if the appointment exists
            if reference in self.appointments_db and self.appointments_db[reference]['user_name'] == user_name:
                appointment = self.appointments_db[reference]
                date = appointment['date']
                time = appointment['time']

                # Remove from MongoDB if available
                if hasattr(self, 'mongo') and self.mongo:
                    self.mongo.db.appointments.delete_one({'reference': reference})

                # Remove from local database
                del self.appointments_db[reference]

                # Add the time slot back to available slots
                if date in self.available_slots:
                    self.available_slots[date].append(time)
                    self.available_slots[date].sort()  # Optional: Sort the available slots

                return {
                    'message': f"Appointment with reference number {reference} has been successfully canceled.",
                    'success': True
                }
            else:
                return {
                    'message': "Sorry, no appointment found with the given reference number.",
                    'success': False
                }
        except Exception as e:
            print(f"Error removing appointment: {str(e)}")
            return {
                'message': f"Sorry, there was an error canceling your appointment: {str(e)}",
                'error': True
            }

    def cancel_appointment(self, reference):
        try:
            if reference in self.appointments_db:
                appointment = self.appointments_db[reference]
                date = appointment['date']
                time = appointment['time']
                self.available_slots[date].append(time)
                del self.appointments_db[reference]
                return "Your appointment has been successfully cancelled."
            return "Sorry, I couldn't find an appointment with that reference number."
        except Exception as e:
            return f"Error cancelling appointment: {str(e)}"

    def check_availability(self, date=None):
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if date in self.available_slots and self.available_slots[date]:
            slots = sorted(self.available_slots[date])
            return f"Available slots for {date}:\n" + "\n".join(slots)
        return "No available slots for this date."

    def get_appointment_prediction(self):
        if not self.model_loaded:
            return "Sorry, prediction service is currently unavailable."
        
        try:
            # You would normally process proper input features here
            prediction = self.prediction_model.predict(np.array([[0]]))
            predicted_appointments = int(prediction[0][0])
            
            if predicted_appointments > 20:
                busy_level = "very busy"
            elif predicted_appointments > 10:
                busy_level = "moderately busy"
            else:
                busy_level = "not very busy"
            
            return f"Based on our analysis, we expect approximately {predicted_appointments} appointments. " \
                   f"It's likely to be {busy_level}."
        except Exception as e:
            return f"Error making prediction: {str(e)}"

    def preprocess_text(self, text):
        # Enhanced text preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_intent(self, text):
        text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(text_vector, self.vectors)[0]
        
        if np.max(similarities) > 0.3:
            most_similar_index = np.argmax(similarities)
            return self.labels[most_similar_index]
        return 'default'

    def generate_response(self, text):
        try:
            intent = self.get_intent(text)
            entities = self._extract_entities(text)
            
            # Log the interaction attempt
            if hasattr(self, 'mongo') and self.mongo and self.current_user:
                self.mongo.db.chat_logs.insert_one({
                    'user_id': self.current_user.get('user_id'),
                    'user_name': self.current_user.get('user_name'),
                    'text': text,
                    'intent': intent,
                    'entities': str(entities),
                    'timestamp': datetime.now()
                })

            if self.current_booking['in_progress']:
                response = self._handle_booking_continuation(text, entities)
            else:
                if intent == 'book_appointment':
                    response = self._handle_booking_intent(entities, text)
                elif intent == 'cancel_appointment':
                    response = self._handle_cancellation_intent(entities)
                elif intent == 'check_availability':
                    response = self._handle_availability_intent(entities)
                elif intent == 'greet':
                    response = self._handle_greeting()
                elif intent == 'goodbye':
                    response = self._handle_goodbye()
                else:
                    response = self._handle_default_intent()

            # Handle different response formats
            if isinstance(response, dict):
                return response.get('message', "I'm not sure how to respond to that.")
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    def _handle_greeting(self):
        user_name = self.current_user.get('user_name', 'there') if self.current_user else 'there'
        return f"Hello {user_name}! How can I help you today? You can book an appointment, check availability, or cancel an existing appointment."

    def _handle_goodbye(self):
        user_name = self.current_user.get('user_name', '') if self.current_user else ''
        return f"Goodbye{' ' + user_name if user_name else ''}! Have a great day!"

    def _handle_default_intent(self):
        return "I'm not sure how to help with that. Would you like to book an appointment or check availability?"
    
    def _handle_booking_intent(self, entities, text):
        try:
            # Extract date from entities or text
            date = entities.get('date')
            
            if not date:
                self.current_booking['in_progress'] = True
                return "When would you like to book your appointment? Please specify a date (e.g., 'tomorrow', 'next Monday')."
            
            # If we have a date, check availability
            self.current_booking = {
                'date': date,
                'time': None,
                'in_progress': True
            }
            
            available_slots = self.check_availability(date)
            return f"What time would you prefer on {date}?\n{available_slots}"
            
        except Exception as e:
            print(f"Error in booking intent: {str(e)}")
            self.current_booking['in_progress'] = False
            return "Sorry, I encountered an error while processing your booking request. Please try again."

    def _handle_booking_continuation(self, text, entities):
        try:
            if not self.current_booking['date']:
                # Handle date selection
                date = entities.get('date')
                if date:
                    self.current_booking['date'] = date
                    available_slots = self.check_availability(date)
                    return f"What time would you prefer on {date}?\n{available_slots}"
                else:
                    return "I didn't catch the date. Please specify when you'd like to book the appointment."
            
            elif not self.current_booking['time']:
                # Handle time selection
                time_match = re.match(r'^([0-9]{1,2}):([0-9]{2})$', text.strip())
                if time_match:
                    requested_time = text.strip()
                    if requested_time in self.available_slots[self.current_booking['date']]:
                        return self.set_appointment(
                            self.current_booking['date'],
                            requested_time,
                            text,
                            self.current_user.get('user_id') if self.current_user else None,
                            self.current_user.get('user_name') if self.current_user else None
                        )
                    else:
                        return f"Sorry, {requested_time} is not available. Please choose from:\n" + \
                            self.check_availability(self.current_booking['date'])
                else:
                    return "Please specify a time in 24-hour format (e.g., '14:30' or '09:00')"
                    
        except Exception as e:
            print(f"Error in booking continuation: {str(e)}")
            self.current_booking['in_progress'] = False
            return "Sorry, I encountered an error. Please start your booking again."

    def _handle_cancellation_intent(self, entities):
        try:
            reference = entities.get('reference')
            if reference:
                return self.cancel_appointment(reference)
            return "Please provide your appointment reference number (format: REFxxxxxx)"
        except Exception as e:
            print(f"Error in cancellation: {str(e)}")
            return "Sorry, I couldn't process your cancellation request. Please try again."

    def _handle_availability_intent(self, entities):
        try:
            date = entities.get('date')
            return self.check_availability(date)
        except Exception as e:
            print(f"Error checking availability: {str(e)}")
            return "Sorry, I couldn't check the availability. Please try again."

    def get_default_response(self, intent):
        responses = {
            'greet': [
                "Hello! How can I help you with your appointment today?",
                "Hi there! Welcome to our GP services. How may I assist you?",
                "Welcome! What can I do for you today?"
            ],
            'goodbye': [
                "Goodbye! Take care!",
                "Have a great day! Come back if you need anything.",
                "Thank you for using our service. Goodbye!"
            ],
            'default': [
                "I'm not sure I understand. Could you please rephrase that?",
                "Could you provide more details about what you need?",
                "I didn't quite catch that. Could you say it differently?"
            ]
        }
        return random.choice(responses.get(intent, responses['default']))

    def chat(self):
        print("Bot: Welcome to GP Services! How can I assist you today?")
        print("Bot: I can help you with:")
        print("     1. Booking appointments")
        print("     2. Checking availability")
        print("     3. Cancelling appointments")
        print("     4. Checking appointment predictions")
        print("     (Type 'quit' to exit)")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Thank you for using our service. Have a great day!")
                break
            
            response = self.generate_response(user_input)
            print("Bot:", response)

def main():
    chatbot = MedicalChatbot()
    chatbot.chat()

if __name__ == "__main__":
    main()