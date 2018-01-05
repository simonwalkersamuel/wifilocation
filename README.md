# wifilocation
A HomeAssistant component for room detection, using machine learning of wifi signals

This is a proof of principle project to see if the wifi signals sampled by an android mobile phone can be used to provide room-level location detection.
It consists of a HomeAssistant sensor component and a Tasker script to run on an Android phone.

Calibration
The first step is to import the Tasker project file (Wifi_Location.ptj,xml) and use it to create a calibration file, using the Sample Wifi task. Go into each room in the house and set the WIFI_ROOM variable to the name of the room (e.g. bedroom, kitchen). Run the task, and move around the room until the sampling is complete. Then, move to the next room, amend the WIFI_ROOM variable, and repeat the calibration step. When all rooms have been sampled, copy the calibration file 

- Set the Tasker HASS_IP variable to "https://<HOME ASSITANT DOMAIN>/api/wifilocation?api_password=<PASS>&job=wifilocation&data="
  
