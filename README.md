# wifilocation
A HomeAssistant component for room detection, using machine learning of wifi signals

This is a proof of principle project to see if the wifi signals sampled by an android mobile phone can be used to provide room-level location detection.
It consists of a HomeAssistant sensor component and a Tasker script to run on an Android phone.

HomeAssitant Installation
Copy wifilocation.py into .homeassitant/custom_components/sensor and add
"sensor:
  - platform: wifilocation"
to the configuration.yaml file.

Calibration
The first step is to import the Tasker project file (Wifi_Location.ptj,xml) and use it to create a calibration file, using the Sample Wifi task. Go into each room in the house and set the WIFI_ROOM variable to the name of the room (e.g. bedroom, kitchen). Run the task, and move around the room until the sampling is complete. Then, move to the next room, amend the WIFI_ROOM variable, and repeat the calibration step.

Machine Learning
Copy the calibration file (Tasker/WifiLocation/wifilocation_calibration.txt).
- Set the Tasker HASS_IP variable to "https://<HOME ASSITANT DOMAIN>/api/wifilocation?api_password=<PASS>&job=wifilocation&data="
  
