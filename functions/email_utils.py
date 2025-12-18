import logging
from email.mime.multipart import MIMEMultipart # lib
from email.mime.text import MIMEText # lib
import smtplib # lib

# Email configuration
GMAIL_USER = "generationofdevices@gmail.com"
GMAIL_PASSWORD = "jyswvahgnltnhave"
RECIPIENT_EMAIL = "generationofdevices@gmail.com"


def send_email(subject: str, body: str, recipient: str = None) -> bool:
    """
    Send email with custom subject and body.
    
    Args:
        subject: Email subject
        body: Email body text
        recipient: Optional recipient (defaults to RECIPIENT_EMAIL)
        
    Returns:
        True if sent successfully, False otherwise
    """
    recipient = recipient or RECIPIENT_EMAIL
    
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, recipient, msg.as_string())

        logging.info(f"Email sent: {subject}")
        return True

    except smtplib.SMTPException as smtp_error:
        logging.error(f"SMTP error: {smtp_error}")
        return False
    except Exception as e:
        logging.error(f"Email error: {e}")
        return False


def send_email_with_error_report(error_message):
    """
    Send error report via email.
    """
    return send_email("Error Report", f"An error occurred during script execution:\n\n{error_message}")


def send_profit_dict_update(symbol: str, interval: str, status: str, 
                            duration_seconds: float = None, trade_count: int = None,
                            error_message: str = None):
    """
    Send profit dict update notification.
    
    Args:
        symbol: Symbol name
        interval: Interval processed
        status: 'completed' or 'error'
        duration_seconds: Time taken in seconds
        trade_count: Number of trades processed
        error_message: Error details if status is 'error'
    """
    if status == 'completed':
        subject = f"✅ Profit Dict: {symbol} {interval} Complete"
        body = f"""Profit Dict Update Completed

Symbol: {symbol}
Interval: {interval}
Duration: {duration_seconds:.2f} seconds
Trades: {trade_count}
"""
    else:
        subject = f"❌ Profit Dict: {symbol} {interval} Error"
        body = f"""Profit Dict Update Error

Symbol: {symbol}
Interval: {interval}
Error: {error_message}
"""
    
    return send_email(subject, body)

