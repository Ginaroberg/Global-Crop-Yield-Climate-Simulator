Here are the steps needed in order to run the demo version of the website locally on your own computer. Please note that the current '
interacitve implementation is slightly laggy, and that future versions will not require .nc file downloads but will instead run custom 
models.

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```
2. Download the zipped .nc files at https://drive.google.com/file/d/18LDvveHPB2UKAlJX0PS-6Wbe0Qa4eA8w/view?usp=drive_link and unzip them
   ```
   Files can be found at this above, and then click extract all in the Repository folder
   ```
3. Start the FastAPI backend server in the same project folder
   ```
    $ uvicorn backend:app
   ```
   
4. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
