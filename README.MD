# Nutrisi OCR Flask App

This project is a Flask application for Optical Character Recognition (OCR) to extract nutritional information from images.

## Features

- Upload images containing nutritional information.
- Extract and display nutritional data from the images.
- Simple and user-friendly web interface.

## Requirements

- Python 3.8+
- Flask
- OpenCV
- Tesseract OCR

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/nutrisi-ocr-flask-app.git
    cd nutrisi-ocr-flask-app
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Install Tesseract OCR:
    - On Ubuntu:
        ```bash
        sudo apt-get install tesseract-ocr
        ```
    - On Windows, download the installer from [here](https://github.com/tesseract-ocr/tesseract/wiki).

## Usage

1. Run the Flask application:
    ```bash
    flask run
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Upload an image containing nutritional information and get the extracted data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
