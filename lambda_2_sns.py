import json
import os
import boto3

# Initialize SNS client
sns_client = boto3.client('sns')

# Get SNS Topic ARN from Environment Variable
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', 'arn:aws:sns:region:acct:ev-notifications-topic')

def lambda_handler(event, context):
    try:
        # Extract payload passed from Lambda 1
        customer_payload = event.get('customer_payload', {})
        decision = event.get('decision', {})
        
        salary = customer_payload.get('salary')
        cibil = customer_payload.get('cibil')
        vehicle_model = customer_payload.get('vehicle_model')
        
        status = decision.get('status')
        confidence = decision.get('confidence_score')
        
        email_subject = f"Update: Your EV Purchase Request for {vehicle_model}"
        
        # Build dynamic email template
        if status == "Approved":
            email_body = (
                f"Great news! Your request for the {vehicle_model} has been approved automatically.\n"
                f"Profile Match Certainty: {confidence * 100}%\n"
            )
        elif status == "Under Review":
            email_body = (
                f"Your application for the {vehicle_model} is under manual review.\n"
                f"System Certainty: {confidence * 100}%\n"
                f"An executive will call you to verify your salary (₹{salary}) and CIBIL score ({cibil})."
            )
        else:
            email_body = (
                f"Thank you for your interest. Regrettably, we cannot approve your purchase profile "
                f"for the {vehicle_model} at this time (Certainty: {confidence * 100}%)."
            )
            
        # Publish message to SNS Topic
        sns_response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=email_subject,
            Message=email_body
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({'MessageId': sns_response['MessageId']})
        }
        
    except Exception as e:
        print(f"Error in Lambda 2 pipeline: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': 'SNS Publish Failed'})}
