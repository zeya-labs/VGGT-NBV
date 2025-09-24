import warnings

# 暂时将 RequestsDependencyWarning 提升为错误，以便我们能看到完整的堆栈跟踪
warnings.simplefilter("error", category=Warning) 

print("Attempting to import requests...")
try:
    import requests
    print("Successfully imported requests.")
    print("requests version:", requests.__version__)
except Exception as e:
    print("\n--- FAILED TO IMPORT OR INITIALIZE REQUESTS ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    print("--------------------------------------------------")