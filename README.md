# decode-labs-project3-customer-segmentation
Overview

SegmentIQ is an end-to-end customer segmentation system built for Project 3 of the Decode Labs Internship.

Businesses rarely serve one homogeneous audience — they serve several distinct groups hidden inside the same dataset. SegmentIQ finds those groups without any labelled training data, using Principal Component Analysis (PCA) to compress correlated features into their essential structure and K-Means clustering to separate customers into naturally occurring segments.

The result is not just a scatter plot. Each mathematical cluster is translated into a named business persona with a spending profile and a recommended marketing action, then delivered through an interactive Streamlit dashboard so non-technical stakeholders can explore the findings themselves.

The core idea: a cluster label is worthless to a marketing manager. A persona with a strategy attached is not.

 Technical Workflow

The pipeline runs in four validated stages.

1️⃣ Data Preprocessing
Missing-value detection and handling across all columns
Categorical encoding of the Gender field into numeric form
Feature scaling with StandardScaler, so that income (measured in thousands) and spending score (1–100) contribute equally to distance calculations

⚠️ Scaling is not optional here — K-Means is a distance-based algorithm, so unscaled features would let annual income dominate the clustering entirely.

2️⃣ Dimensionality Reduction (PCA)
PCA applied to the scaled feature matrix
Components retained to preserve 95% of the total variance
Removes multicollinearity and noise while keeping the signal intact
Produces a compact space suitable for both clustering and 2D/3D visualisation
3️⃣ Clustering (K-Means)
K-Means fitted across a range of candidate cluster counts
Cluster assignments computed on the PCA-transformed data
4️⃣ Validation — Choosing K Properly

The number of clusters is proven, not assumed, using two independent methods:

Method	What it measures	Why it matters
Elbow Method	Within-cluster sum of squares (inertia) vs. K	Identifies where additional clusters stop meaningfully reducing error
Silhouette Score	Cohesion vs. separation, per sample	Confirms clusters are genuinely distinct, not just numerous

Both methods converge on K = [insert your optimal K], giving a defensible, evidence-backed segmentation.

5️⃣ Persona Creation

Each cluster's centroid is profiled across age, income and spending behaviour, then translated into a business-readable persona — for example, a high-income / low-spending group representing untapped revenue potential. Every persona card carries its size, defining traits, and a suggested engagement strategy.

 The Interactive Dashboard

Built with Streamlit, the dashboard is designed for the person who needs the answer, not the notebook.

What a manager can do without writing a single line of code:

🔍 Explore every segment through interactive 2D and 3D cluster plots
📈 Inspect the validation evidence — elbow curve and silhouette plots rendered live
🧾 Read the persona cards — each cluster summarised in plain business language
🎚️ Filter and drill down into individual customer records by segment
⬇️ Export results — download the segmented dataset for use in CRM or campaign tools
📱 Use it anywhere — fully responsive layout with a clean light theme

The interface uses a consistent teal design system (
#AFEEEE → 
#20B2AA) with animated transitions and hover states, so it is presentation-ready in front of management rather than looking like a debug tool.

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
│   └── personas.py         # Cluster → business persona mapping
│
├── data/
│   └── Mall_Customers.csv  # Source dataset
│
├── assets/                 # Screenshots and static files
├── requirements.txt
└── README.md
⚙️ Installation & Usage
Prerequisites
Python 3.9 or higher
pip
1. Clone the repository
bash
git clone https://github.com/Yalda-Ashrafi/segmentiq.git
cd segmentiq
2. Create and activate a virtual environment

Windows (PowerShell)

powershell
python -m venv venv
venv\Scripts\activate

macOS / Linux

bash
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the pipeline (optional)

Generates the trained model artefacts and clustered output.

bash
python run_pipeline.py
5. Launch the dashboard
bash
streamlit run app.py

The app opens automatically at http://localhost:8501.

 Example Output

Screenshots to be added.

Preview	Description
![Dashboard Home](assets/dashboard-home.png)	Landing view with dataset summary and key metrics
![Elbow & Silhouette](assets/validation.png)	Elbow curve and silhouette analysis used to select K
![Cluster Plot](assets/clusters.png)	2D / 3D PCA projection with colour-coded segments
![Persona Cards](assets/personas.png)	Business personas with traits and recommended actions

Replace the placeholder paths above with your captured images in assets/.

🛠️ Technologies Used
Technology	Role in the project
Python	Core language for the entire pipeline
pandas	Data loading, cleaning and transformation
NumPy	Numerical operations and array handling
scikit-learn	StandardScaler, PCA, KMeans, silhouette metrics
Matplotlib	Elbow curves and static visualisations
Seaborn	Statistical plots and distribution analysis
Streamlit	Interactive dashboard and deployment layer
 Acknowledgment

This project was developed as Project 3 of the Decode Labs Internship — the unsupervised learning module of the program.

My role: sole developer. I designed and implemented the complete solution end to end — data preprocessing, PCA implementation, K-Means clustering with elbow and silhouette validation, the cluster-to-persona translation logic, the modular code architecture, and the full Streamlit dashboard interface.

Thank you to the Decode Labs team for the project brief and technical guidance throughout the internship.

 Conclusion

SegmentIQ demonstrates the step that separates a machine learning exercise from a business tool: translation.

Finding clusters is a solved technical problem. Making those clusters legible — so a marketing lead can see that one segment is under-served, another is over-invested in, and act on it the same afternoon — is where the value is created. By pairing statistically validated segmentation with an interface that requires no coding, SegmentIQ puts data-driven customer strategy directly in the hands of the people who make those decisions.

Complexity in. Clarity out.
