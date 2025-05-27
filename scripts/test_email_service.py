import asyncio
import sys
from backend.services.notification_service import notification_service
from backend.db.mongodb.models.notification import NotificationType, NotificationPriority

async def test_email_notification():
    """Test sending different types of email notifications."""
    
    # Test cases
    test_cases = [
        {
            "name": "Task Reminder",
            "template_name": "test_notification",
            "template_data": {
                "subject": "Task Reminder",
                "user_name": "Test User",
                "message": "You have a task due tomorrow: Review Project Proposal",
                "action_url": "https://lumicoria.ai/tasks/123",
                "year": "2024"
            },
            "priority": NotificationPriority.HIGH
        },
        {
            "name": "Document Processed",
            "template_name": "test_notification",
            "template_data": {
                "subject": "Document Processed",
                "user_name": "Test User",
                "message": "Your document 'Project_Proposal.pdf' has been processed successfully.",
                "action_url": "https://lumicoria.ai/documents/456",
                "year": "2024"
            },
            "priority": NotificationPriority.NORMAL
        },
        {
            "name": "Wellbeing Alert",
            "template_name": "test_notification",
            "template_data": {
                "subject": "Time for a Break",
                "user_name": "Test User",
                "message": "You've been working for 50 minutes. Time to take a short break!",
                "action_url": "https://lumicoria.ai/wellbeing/break",
                "year": "2024"
            },
            "priority": NotificationPriority.LOW
        }
    ]

    # Your test email address
    test_email = "your-email@example.com"  # Replace with your email

    print(f"Testing email notifications...")
    print(f"Sending to: {test_email}")
    print("-" * 50)

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            success = await notification_service.send_email_notification(
                to_email=test_email,
                template_name=test_case["template_name"],
                template_data=test_case["template_data"],
                priority=test_case["priority"]
            )
            if success:
                print("✅ Email sent successfully!")
            else:
                print("❌ Failed to send email")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        print("-" * 50)

if __name__ == "__main__":
    # Get email from command line argument if provided
    if len(sys.argv) > 1:
        test_email = sys.argv[1]
    else:
        test_email = input("Enter your email address for testing: ")

    # Run the test
    asyncio.run(test_email_notification()) 