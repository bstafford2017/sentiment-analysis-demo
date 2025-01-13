import tensorflow as tf
import base64

categories = 90
batch_size = 32
img_height = 180
img_width = 180

def lambda_handler(event, context):

    # Grab byte array from request
    base64_bytes = event.body

    # Check if empty
    if not base64_bytes:
        return {
            "statusCode": 403,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": "Invalid image input"
            })
        }

    # Base 64 decode
    image_bytes = base64_bytes.decode('ascii')

    # Load existing model
    model = tf.keras.models.load_model('src/animal-recognition/model.keras')

    # Perform prediction
    predictions = model.predict(image_bytes)

    # Perform distribution
    score = tf.nn.softmax(predictions[0])

    # Format result
    result = "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "result": result
        })
    }
