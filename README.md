# SegmentIQ: Customer Segmentation with PCA & K-Means
### Decode Labs Internship – Project 3
Unsupervised learning that turns raw customer data into business personas anyone can act on.

![Python](https://github.com/Yalda-Ashrafi/decode-labs-project3-customer-segmentation/blob/b0358166ec8a8e9d7eb7cd59de798d69cb4d2695/assets/1.png)
![scikit-learn](https://github.com/Yalda-Ashrafi/decode-labs-project3-customer-segmentation/blob/b0358166ec8a8e9d7eb7cd59de798d69cb4d2695/assets/2.png)
![Streamlit](https://github.com/Yalda-Ashrafi/decode-labs-project3-customer-segmentation/blob/b0358166ec8a8e9d7eb7cd59de798d69cb4d2695/assets/3.png)
![pandas](https://github.com/Yalda-Ashrafi/decode-labs-project3-customer-segmentation/blob/b0358166ec8a8e9d7eb7cd59de798d69cb4d2695/assets/4.png)
![Status](https://github.com/Yalda-Ashrafi/decode-labs-project3-customer-segmentation/blob/b0358166ec8a8e9d7eb7cd59de798d69cb4d2695/assets/5.png)

> Decode Labs Internship, Project 3

##  Overview

SegmentIQ discovers the distinct customer groups hidden inside a single dataset, without any labelled training data.

The pipeline applies **Principal Component Analysis (PCA)** to compress correlated features into their essential structure, then uses **K-Means clustering** to separate customers into natural segments. The optimal number of clusters is proven with elbow and silhouette validation rather than assumed.

Each mathematical cluster is then translated into a **named business persona** with a spending profile and a recommended marketing action, and delivered through an interactive dashboard that a manager can explore without writing code.

> A cluster label means nothing to a marketing team. A persona with a strategy attached does.

##  Features

* 📊 **Interactive Streamlit dashboard** for exploring every segment visually
* 📈 **Elbow and silhouette validation plots** rendered live inside the app
* 🧾 **Persona cards** translating each cluster into plain business language
* 🔍 **Filter and drill down** into individual customer records by segment
* ⬇️ **Exportable segmented dataset** ready for CRM or campaign tools
* 📱 **Responsive light theme** designed to be presented to management

## How It Works

### Step 1: Data Preprocessing

Missing values are handled, the `Gender` column is encoded into numeric form, and all features are scaled with `StandardScaler`.

Scaling is essential here. K-Means measures distance, so without it annual income in thousands would completely overpower spending score on a 1 to 100 range.

### Step 2: Dimensionality Reduction

PCA is applied to the scaled feature matrix, retaining enough components to preserve **95% of the total variance**. This removes multicollinearity and noise while keeping the real signal intact.

### Step 3: Clustering

K-Means is fitted across a range of candidate cluster counts on the PCA transformed data.

### Step 4: Validation

Two independent methods confirm the right number of clusters.

| Method | What It Measures | Why It Matters |
| :--- | :--- | :--- |
| Elbow Method | Within cluster sum of squares against K | Finds where extra clusters stop reducing error |
| Silhouette Score | Cluster cohesion against separation | Confirms segments are genuinely distinct |

Both methods agree on **K = [insert your optimal K]**.

### Step 5: Persona Creation

Each cluster centroid is profiled across age, income and spending behaviour, then mapped to a business readable persona with its segment size, defining traits, and a suggested engagement strategy.

## 📁 Project Structure

| Folder / File | Description |
| :--- | :--- |
| `app.py` | Streamlit dashboard entry point |
| `run_pipeline.py` | Runs the full ML pipeline end to end |
| `src/preprocessing.py` | Data cleaning, encoding and scaling |
| `src/dimensionality.py` | PCA with 95% variance retention |
| `src/clustering.py` | K-Means, elbow method, silhouette score |
| `src/personas.py` | Cluster to persona mapping |
| `data/Mall_Customers.csv` | Source dataset |
| `assets/` | Screenshots and static files |
| `requirements.txt` | Project dependencies |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yalda-Ashrafi/segmentiq.git
cd segmentiq
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it

Windows:

```bash
venv\Scripts\activate
```

macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

##  Usage

### Run the full pipeline

Generates the trained model artefacts and the clustered output.

```bash
python run_pipeline.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

##  Example Output

Screenshots to be added.
![Status](https://github.com/Yalda-Ashrafi/decode-labs-project3-customer-segmentation/blob/78d085f464622e911bc5940f9716cebb1b3058ab/assets/6.png)


### Dashboard Home

Landing view with the dataset summary and key metrics.

### Validation Plots

The elbow curve and silhouette analysis used to select the optimal K.

### Cluster Visualisation

PCA projection with colour coded segments in 2D and 3D.

### Persona Cards

Each segment summarised with its traits and recommended business action.

## Dependencies

| Library | Role in the Project |
| :--- | :--- |
| `python` | Core language, version 3.9 or higher |
| `pandas` | Data loading, cleaning and transformation |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | StandardScaler, PCA, KMeans, silhouette metrics |
| `matplotlib` | Elbow curves and static visualisations |
| `seaborn` | Statistical plots and distribution analysis |
| `streamlit` | Interactive dashboard and deployment layer |

##  Acknowledgment

This project was developed as Project 3 of the **Decode Labs Internship**, covering the unsupervised learning module of the program.

### My Role

Sole developer. I designed and built the complete solution end to end, including the preprocessing pipeline, the PCA implementation, K-Means clustering with elbow and silhouette validation, the cluster to persona translation logic, the modular code architecture, and the full Streamlit dashboard interface.

Thank you to the Decode Labs team for the project brief and technical guidance throughout the internship.

##  Live Demo
Explore the interactive dashboard here: https://customer-segmentation-yalda-ashrafi.streamlit.app/

##  Project Video
Watch the walkthrough video: [SegmentIQ Demo Video](https://drive.google.com/file/d/13elR05IHz748RgVklcIP4w9_sOhj5Sen/view?usp=sharing)

##  Conclusion

Finding clusters is a solved technical problem. Making those clusters legible, so a marketing lead can see which segment is under served and act on it the same afternoon, is where the value is created.

By pairing statistically validated segmentation with an interface that requires no coding, SegmentIQ puts customer strategy directly in the hands of the people who make those decisions.

**Complexity in. Clarity out.**

## 👩‍💻 Author

Built by **Yalda Ashrafi** for the Decode Labs Internship 2026.

If you found this useful, consider giving the repo a ⭐
