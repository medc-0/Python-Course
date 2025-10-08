"""
05-email-sender.py

Sending emails with Python.

Overview:
---------
Automate sending emails using Python. Since 2022, Gmail and many other providers require app passwords or OAuth for SMTP access.
For beginners, using app passwords is the easiest way with Gmail. Alternatively, you can use third-party services like SendGrid, Mailgun, or SMTP2GO, which provide APIs and easier authentication.

Examples (SMTP with app password):
----------------------------------
"""

import smtplib

sender = "your_email@gmail.com"
receiver = "receiver_email@gmail.com"
password = "your_app_password"  # Generate an app password in your Google Account settings

message = """Subject: Test Email

This is an automated email from Python."""

# Uncomment and fill in your credentials to use
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(sender, password)
    server.sendmail(sender, receiver, message)
print("Email sent!")

"""
Alternative: Using SendGrid (API-based, recommended for production)
-------------------------------------------------------------------
# pip install sendgrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

message = Mail(
    from_email='your_email@example.com',
    to_emails='receiver@example.com',
    subject='Test Email',
    plain_text_content='This is an automated email from Python.'
)
sg = SendGridAPIClient('YOUR_SENDGRID_API_KEY')
response = sg.send(message)
print(response.status_code)

Tips:
-----
- For Gmail, generate an app password: https://support.google.com/accounts/answer/185833
- For SendGrid, see official docs: https://docs.sendgrid.com/for-developers/sending-email/api-getting-started
- For Mailgun, see: https://documentation.mailgun.com/en/latest/quickstart-sending.html
- Never share your real password or API key in code.
- Use environment variables or config files for credentials.
- For more on smtplib: https://docs.python.org/3/library/smtplib.html
"""
