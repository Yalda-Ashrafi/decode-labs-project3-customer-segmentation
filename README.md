# decode-labs-project3-customer-segmentation

SegmentIQ

Customer Segmentation with Unsupervised Learning

Turning raw customer data into business personas that managers can actually act on.

Show Image Show Image Show Image Show Image Show Image

📌 Overview

SegmentIQ is an end to end customer segmentation system built for Project 3 of the Decode Labs Internship.

Businesses rarely serve one single audience. They serve several distinct groups hidden inside the same dataset. SegmentIQ finds those groups without any labelled data, using Principal Component Analysis (PCA) to compress the features down to their essential structure, and K-Means clustering to separate customers into natural segments.

The result is not just a scatter plot. Every mathematical cluster is translated into a named business persona with a spending profile and a recommended marketing action, then delivered through an interactive Streamlit dashboard that anyone can explore.

A cluster label means nothing to a marketing manager. A persona with a strategy attached does.

🧠 Technical Workflow
Step 1: Data Preprocessing
Missing values detected and handled across all columns
The Gender column encoded into numeric form
All features scaled with StandardScaler

Scaling matters here. K-Means measures distance, so without it annual income (in thousands) would completely overpower spending score (1 to 100) and the clusters would be meaningless.

Step 2: Dimensionality Reduction with PCA
PCA applied to the scaled feature matrix
Components retained to preserve 95% of the total variance
Removes multicollinearity and noise while keeping the real signal
Produces a compact space suitable for both clustering and visualisation
Step 3: K-Means Clustering
K-Means fitted across a range of candidate cluster counts
Cluster assignments computed on the PCA transformed data
Step 4: Validating the Number of Clusters

The number of clusters is proven, not assumed. Two independent methods were used.

Elbow Method

Plots the within cluster sum of squares against K, and shows the point where adding more clusters stops meaningfully reducing error.

Silhouette Score

Measures how tight each cluster is compared to how far apart the clusters are, confirming the segments are genuinely distinct rather than just numerous.

Both methods agree on K = [insert your optimal K], giving a segmentation that can be defended with evidence.

Step 5: Persona Creation

Each cluster centroid is profiled across age, income and spending behaviour, then translated into a business readable persona. Every persona card shows the size of the segment, its defining traits, and a suggested engagement strategy.

📊 The Interactive Dashboard

Built with Streamlit and designed for the person who needs the answer, not the notebook.

What a Manager Can Do Without Writing Code
🔍 Explore every segment through interactive 2D and 3D cluster plots
📈 Inspect the validation evidence, with the elbow curve and silhouette plots rendered live
🧾 Read the persona cards, each cluster summarised in plain business language
🎚️ Filter and drill down into individual customer records by segment
⬇️ Export the segmented dataset for use in CRM or campaign tools
📱 Use it on any screen size, with a clean light theme

The interface uses a consistent teal design system with animated transitions and hover states, so it is ready to present to management rather than looking like a debug tool.

🗂️ Project Structure
SegmentIQ/
│
├── app.py                  # Streamlit dashboard entry point
├── run_pipeline.py         # Runs the full ML pipeline end to end
│
├── modules/
│   ├── preprocessing.py    # Cleaning, encoding, scaling
│   ├── dimensionality.py   # PCA with 95% variance retention
│   ├── clustering.py       # K-Means, elbow, silhouette
│   └── personas.py         # Cluster to business persona mapping
│
├── data/
│   └── Mall_Customers.csv  # Source dataset
│
├── assets/                 # Screenshots and static files
├── requirements.txt
└── README.md
⚙️ Installation and Usage
Prerequisites
Python 3.9 or higher
pip
1. Clone the Repository
bash
git clone https://github.com/Yalda-Ashrafi/segmentiq.git
cd segmentiq
2. Create and Activate a Virtual Environment

Windows (PowerShell):

powershell
python -m venv venv
venv\Scripts\activate

macOS or Linux:

bash
python3 -m venv venv
source venv/bin/activate
3. Install the Dependencies
bash
pip install -r requirements.txt
4. Run the Pipeline

This generates the trained model artefacts and the clustered output.

bash
python run_pipeline.py
5. Launch the Dashboard
bash
streamlit run app.py

The app opens automatically at http://localhost:8501

🖼️ Example Output

Screenshots to be added.

Dashboard Home

Landing view with the dataset summary and key metrics.

Elbow and Silhouette Validation

The two plots used to select the optimal number of clusters.

Cluster Visualisation

PCA projection with colour coded segments in 2D and 3D.

Persona Cards

Business personas with their traits and recommended actions.

🛠️ Technologies Used
Python for the core language of the entire pipeline
pandas for data loading, cleaning and transformation
NumPy for numerical operations and array handling
scikit-learn for StandardScaler, PCA, KMeans and silhouette metrics
Matplotlib for elbow curves and static visualisations
Seaborn for statistical plots and distribution analysis
Streamlit for the interactive dashboard and deployment layer
🎓 Acknowledgment

This project was developed as Project 3 of the Decode Labs Internship, the unsupervised learning module of the program.

My Role

Sole developer. I designed and built the complete solution end to end, including the data preprocessing, the PCA implementation, K-Means clustering with elbow and silhouette validation, the cluster to persona translation logic, the modular code architecture, and the full Streamlit dashboard interface.

Thank you to the Decode Labs team for the project brief and the technical guidance throughout the internship.

🚀 Conclusion

SegmentIQ demonstrates the step that separates a machine learning exercise from a business tool, which is translation.

Finding clusters is a solved technical problem. Making those clusters legible, so that a marketing lead can see which segment is under served and act on it the same afternoon, is where the value is created. By pairing statistically validated segmentation with an interface that requires no coding, SegmentIQ puts customer strategy directly in the hands of the people who make those decisions.

Complexity in. Clarity out.

👩‍💻 Author

Built by Yalda Ashrafi for the Decode Labs Internship 2026.

If you found this useful, consider giving the repo a ⭐
