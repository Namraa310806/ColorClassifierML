# NP's Color Classifier 🎨

A sophisticated color analysis tool that uses machine learning to identify and analyze the distribution of colors in images. This application provides a detailed breakdown of color composition in any uploaded image.

## 🌟 Features

- **Smart Color Analysis**: Uses machine learning to accurately identify and classify colors
- **Real-time Processing**: Instant color distribution analysis
- **Visual Color Swatches**: See the actual colors alongside their percentages
- **User-friendly Interface**: Clean and intuitive design
- **High Accuracy**: Trained on a diverse dataset of color samples
- **Support for Multiple Formats**: Works with JPG, JPEG, and PNG images

## 📸 Screenshots

The application interface can be found in the `screenshots` folder:
- `main_interface.png`: Main application interface
- `color_analysis.png`: Example of color analysis results
- `upload_example.png`: Image upload and processing example

## 🛠️ Technical Details

### Technologies Used
- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn
- **Image Processing**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Color Analysis**: Matplotlib

### Model Architecture
- Random Forest Classifier
- Color histogram extraction
- HSV color space analysis
- Patch-based color distribution

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/color-classifier.git
cd color-classifier
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
color-classifier/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── ColorClassification_dataset/  # Training dataset
│   ├── red/              # Red color samples
│   ├── blue/             # Blue color samples
│   └── ...               # Other color samples
├── uploads/              # Temporary storage for uploaded images
└── screenshots/          # Application screenshots
    ├── main_interface.png
    ├── color_analysis.png
    └── upload_example.png
```

## 💡 How It Works

1. **Image Upload**: Users can upload any JPG, JPEG, or PNG image
2. **Color Extraction**: The application extracts color histograms from the image
3. **Analysis**: The machine learning model analyzes the color distribution
4. **Results**: A detailed breakdown of colors is displayed with percentages

## 🎯 Model Performance

- **Accuracy**: ~99.3% on test dataset
- **Color Categories**: 9 distinct color classes
- **Processing Time**: Near real-time analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Namraa Patel**
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- scikit-learn community for the machine learning tools
- OpenCV team for image processing capabilities 