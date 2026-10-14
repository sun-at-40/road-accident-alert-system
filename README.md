





https://github.com/user-attachments/assets/69ecddbf-3db9-43bb-9c0e-81497fb50600






# Road Accident Detection & Alert System

# Overview

### The Road Accident Detection & Alert System is a project designed to detect accidents on roads, trigger alerts, capture accident details, and enable emergency response. It utilizes computer vision, audio alerts, and communication technologies to improve response times and assist in emergency situations.

# Features

### Accident Detection: Utilizes computer vision algorithms to detect accidents based on visual cues.

### Alert System: Triggers an alert sound upon detecting an accident to notify nearby individuals.

### Screenshot Capture: Captures a screenshot of the accident scene along with date and time for documentation.

### Emergency Response: Allows authorized personnel to initiate emergency medical services by calling an ambulance.

# Technologies Used

### Computer Vision: OpenCV for image processing and object detection.

### Audio Alert: Integration with sound libraries for alert notifications.

### Communication: Integration with twilio for emergency response.

### Documentation: GitHub for version control and project sharing.

# Getting Started

## To use the Road Accident Detection & Alert System, follow these steps:

### Clone the repository to your local machine.
### Install the necessary dependencies listed in the requirements file.
### Configure the system settings and authorized personnel details.
### Run the main application script to start monitoring for accidents.
### In case of an accident, follow the on-screen instructions for emergency response actions.

# Usage
### The system is designed for use in surveillance and monitoring scenarios where quick detection and response to accidents are crucial. It can be deployed in traffic management systems, surveillance cameras, and other relevant environments.

# Authentication

The web application requires a user to sign in before viewing reports, searching vehicles, or analysing videos. Users can create an account with an email address and a password of at least eight characters.

## Google sign-in setup

Google sign-in uses OpenID Connect. Create a **Web application** OAuth 2.0 client in Google Cloud, then add these values to `.env` (or provide them as environment variables):

```dotenv
FLASK_SECRET_KEY=replace-with-a-long-random-secret
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
# Set to true when the app is served over HTTPS.
SESSION_COOKIE_SECURE=false
```

In the Google OAuth client configuration, add this authorized redirect URI for local development:

```text
http://127.0.0.1:5000/login/google/authorize
```

Use the deployed application's HTTPS URL with `/login/google/authorize` appended in production. The Google button remains unavailable until both Google credentials are configured. Password hashes and user identities are stored in `instance/users.sqlite3`; this database must be retained when deploying the application.

# Contact
### For Dataset and inquiries, contact [Soham Soni](mailto:sonisoham91@gmail.com) .
