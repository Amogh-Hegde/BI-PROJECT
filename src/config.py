from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REPORT_DIR = PROJECT_ROOT / "report"
PPT_DIR = PROJECT_ROOT / "ppt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
XAI_DIR = OUTPUTS_DIR / "xai"
TABLES_DIR = OUTPUTS_DIR / "tables"
DASHBOARD_ASSETS_DIR = OUTPUTS_DIR / "dashboard_assets"


SEVERITY_ORDER = ["Minor", "Serious", "Fatal"]


STATE_CITY_MAP = {
    "Maharashtra": [("Mumbai", 19.0760, 72.8777), ("Pune", 18.5204, 73.8567), ("Nagpur", 21.1458, 79.0882)],
    "Tamil Nadu": [("Chennai", 13.0827, 80.2707), ("Coimbatore", 11.0168, 76.9558), ("Madurai", 9.9252, 78.1198)],
    "Karnataka": [("Bengaluru", 12.9716, 77.5946), ("Mysuru", 12.2958, 76.6394), ("Hubballi", 15.3647, 75.1240)],
    "Uttar Pradesh": [("Lucknow", 26.8467, 80.9462), ("Kanpur", 26.4499, 80.3319), ("Varanasi", 25.3176, 82.9739)],
    "Delhi": [("New Delhi", 28.6139, 77.2090), ("Dwarka", 28.5921, 77.0460), ("Rohini", 28.7495, 77.0565)],
    "West Bengal": [("Kolkata", 22.5726, 88.3639), ("Howrah", 22.5958, 88.2636), ("Durgapur", 23.5204, 87.3119)],
    "Rajasthan": [("Jaipur", 26.9124, 75.7873), ("Jodhpur", 26.2389, 73.0243), ("Udaipur", 24.5854, 73.7125)],
    "Gujarat": [("Ahmedabad", 23.0225, 72.5714), ("Surat", 21.1702, 72.8311), ("Vadodara", 22.3072, 73.1812)],
    "Telangana": [("Hyderabad", 17.3850, 78.4867), ("Warangal", 17.9689, 79.5941), ("Nizamabad", 18.6725, 78.0941)],
    "Kerala": [("Kochi", 9.9312, 76.2673), ("Thiruvananthapuram", 8.5241, 76.9366), ("Kozhikode", 11.2588, 75.7804)],
}


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
