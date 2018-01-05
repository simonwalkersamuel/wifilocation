# wifilocation
A HomeAssistant component for room detection, using machine learning of wifi signals

This is a proof-of-principle project to see if the wifi signals sampled by an android mobile phone can be used to provide room-level location detection. Using a minimum (probably) of three wifi routers, placed around your house, the signal strength from them (and your neighbours) can be used to determine your location.

WifiLocation consists of a HomeAssistant custom component (a sensor) and a Tasker project that runs on an Android phone. Only Android can be used at the moment. 

The Home Assistant component has the following dependencies for machine leanring (and possibly more...):
- numpy
- scikit
- Keras
- TensorFlow

HomeAssitant Installation

Instructions for instlaling Home Assistant can be found here (https://home-assistant.io/). Once instlaled and up-and-running, copy wifilocation.py into the .homeassitant/custom_components/sensor directory.

Edit the Home Assistant configuraiton.yaml file to include:
sensor:
  - platform: wifilocation
to the configuration.yaml file.

Calibration

Import the Tasker project file (Wifi_Location.ptj,xml) and use it to create a calibration file, using the Sample Wifi task. Go into each room in the house and set the WIFI_ROOM variable to the name of the room (e.g. bedroom, kitchen). Run the task, and move around the room until the sampling is complete. Then, move to the next room, amend the WIFI_ROOM variable, and repeat the calibration step.

Machine Learning

Copy the calibration file from the Android phone (Tasker/WifiLocation/wifilocation_calibration.txt) onto the HomeAssistant server.

Testing
- Set the Tasker HASS_IP variable to "https://<HOME ASSITANT DOMAIN>/api/wifilocation?api_password=<PASS>&job=wifilocation&data="

