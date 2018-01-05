import numpy as np
import os
import logging
from datetime import timedelta
import datetime
import time
import requests
import voluptuous as vol
import asyncio
import json

import homeassistant.helpers.config_validation as cv
from homeassistant.const import (
    CONF_HOST, CONF_PORT, CONF_SCAN_INTERVAL, CONF_RESOURCES)
from homeassistant.util import Throttle
from homeassistant.helpers.entity import Entity
from homeassistant.helpers import intent
from homeassistant.components.climate import (ClimateDevice, PLATFORM_SCHEMA)
from homeassistant.const import (CONF_NAME, CONF_HOST, CONF_PORT,
                                 TEMP_CELSIUS, ATTR_TEMPERATURE)
import homeassistant.helpers.config_validation as cv                                 
from homeassistant.components.http import HomeAssistantView

import requests
import time
import sys

_LOGGER = logging.getLogger(__name__)
_LOGGER.info('called')
 
DOMAIN = 'wifilocation'
DEPENDENCIES = ['http']

MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=5)

class WifiLocationResponse(object):

    def __init__(self, room=None, probability=None):
        """Initialize the Dialogflow response."""
        self.room = room
        self.probability = probability

    def as_dict(self):
        """Return response in a dictionary."""
        return {
            'room': self.speech,
            'probability': self.speech
        }

class WifiLocationView(HomeAssistantView):

    url = '/api/wifilocation'
    name = 'api:wifilocation'

    def __init__(self, add_devices):
        self.control = WifiLocationControl()
        self.add_devices = add_devices
        self.add_devices([self.control])

    @asyncio.coroutine
    def get(self, request):
        _LOGGER.info('WifiLocation GET received')
        res = yield from self.async_handle(request.query)
        #import pdb
        #pdb.set_trace()
        #req = request.query
        #res = self.json({'room':'unknown','prob':'1.00'})
        #if req.query.get('job')=='wifi_location':
        #    sample = req.get('data')
        #    signal, mac = self.control.parse_sample(sample)
        #    user = req.get('user')
        #    room,prob = self.control.predict(mac,signal,normalise=True)
        #    self.control._state = '{} ({:0.2f})'.format(room,prob)
        #    self.control._location = room
        #    self.control._probability = prob
        #    res = self.json({'room':room,'prob':'{:0.2f}'.format(prob)})
        return res

    @asyncio.coroutine
    def async_handle(self, req):
        response = self.json({'room':'unknown','prob':'1.00'})
        if req.get('job')=='wifi_location':
            sample = req.get('data')
            signal, mac = self.control.parse_sample(sample)
            user = req.get('user')
            room,prob = self.control.predict(mac,signal,normalise=True)
            self.control._state = '{} ({:0.2f})'.format(room,prob)
            self.control._location = room
            self.control._probability = prob
            response = self.json({'room':room,'prob':'{:0.2f}'.format(prob)})
        return response

def setup(hass, config):
    hass.states.set('WifiLocationControl.initialised', 'True')
    return True
    
@asyncio.coroutine
def async_setup_platform(hass, config, async_add_devices, discovery_info=None):
    """Set up Thermostat Sensor."""
    hass.http.register_view(WifiLocationView(async_add_devices))

    return True
            
class WifiLocationControl(Entity): 

    def __init__(self, location='Home'):
        self._name = 'WifiLocation'
        self._location = None
        self._state = 'Available'
        self._probability = None

        self.macs = None
        self.rooms = None
        self.ssids = None

        self.SSIDS_TO_USE = None
        self.REFERENCE_MAC = []
        
        self.dir = '/home/simon/.homeassistant/WifiLocation'
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        self.calib_file = os.path.join(self.dir,'wifilocation_calibration.txt')
        self.model_weights_file = os.path.join(self.dir,'wifilocation_weights.txt')
        self.sample_file = os.path.join(self.dir,'scan_results.txt')
        self.read_calibration()

        self.model = self.create_model(nrooms=len(self.rooms),nmacs=len(self.macs))
        self.load_weights()
           
    @property
    def name(self):
        return self._name

    @property
    def location(self):
        return self._location

    @property
    def probability(self):
        return self._probability
        
    @property
    def state(self):
        return self._state
        
    @property    
    def device_state_attributes(self):
        return {'name':self._name,'state':self._state,'location':self._location,'probability':self._probability}
            
    def read_calibration(self,n_samples_per_room=200):
        with open(self.calib_file,'r') as f:
            data = f.read()
        
        data = data.split('[')
        res = []
        macs = []
        ssids = []
        rooms = []
        room_count = []
        for dCur in data:
            spl = dCur.split(']')
        
            if len(spl)==2:
                tmp = spl[0].split(',')
                if len(tmp)==3:
                    date,time,room = tmp
                    date = date.strip()
                    time = time.strip()
                    room = room.strip()
                    if room not in rooms:
                        rooms.append(room)
                        room_count.append(0)
                    
                spl2 = spl[1].split('&&')
                for entry in spl2:
                    spl3 = entry.split('|')
                    mac_list = []
                    sig_list = []
                    ssid_list = []
                    for entry1 in spl3:
                        try:
                            tmp = entry1.split(',')
                            if len(tmp)==3:
                                ssid,signal,mac = tmp
                                ssid = ssid.strip()
                                signal = signal.strip()
                                mac = mac.strip()
                                if self.SSIDS_TO_USE is not None:
                                    if ssid in self.SSIDS_TO_USE:
                                       mac_list.append(mac)
                                       sig_list.append(float(signal))
                                       ssid_list.append(ssid)
                                       if mac not in macs:
                                           macs.append(mac)
                                           ssids.append(ssid)
                                           self.REFERENCE_MAC.append(mac)
                                else:
                                    mac_list.append(mac)
                                    sig_list.append(float(signal))
                                    ssid_list.append(ssid)
                                    if mac not in macs:
                                        macs.append(mac)
                                        ssids.append(ssid)
                                        self.REFERENCE_MAC.append(mac)
                        except:
                            _LOGGER.error(''.format(sys.exc_info()[1]))
                    
                    if len(sig_list)>0 and room_count[rooms.index(room)]<n_samples_per_room:
                        res.append({'room':room,'time':time,'date':date,'ssid':ssid_list,'signal':sig_list,'mac':mac_list})
                        room_count[rooms.index(room)] += 1
                    
        self.macs = macs
        self.rooms = rooms
        self.ssids = ssids
        
        return res

    def parse_sample(self,data):
        if data is None:
            with open(self.sample_file,'r') as f:
                data = f.read()
            
        spl = data.split('|')
        mac_list = []
        sig_list = []
        ssid_list = []
        for entry in spl:
            try:
                tmp = entry.split(',')
                if len(tmp)==3:
                    ssid,signal,mac = tmp
                    ssid = ssid.strip()
                    signal = signal.strip()
                    mac = mac.strip()
                    mac_list.append(mac)
                    sig_list.append(float(signal))
                    ssid_list.append(ssid)
            except:
                print(entry)
                
        #si = self.normalised_signal(sig_list,mac_list,self.macs)
        return sig_list, mac_list
       
    def normalised_signal(self,sig,macs,known_macs):

        si = np.zeros(len(known_macs),dtype='float') #+ float('nan')
        #if self.REFERENCE_MAC not in macs:
        #    return si
        minval = -127.
        try:
            ref_si_inds = np.asarray([macs.index(r) for r in self.REFERENCE_MAC if r in macs])
            if len(ref_si_inds)>0:
                ref_si = np.asarray(sig)[ref_si_inds]
            else:
                ref_si = 0.
            ref_si += minval
        except:
            import pdb
            pdb.set_trace()

        for j,s in enumerate(sig):
                if macs[j] in known_macs:
                    si[known_macs.index(macs[j])] = ((s+minval) / np.mean(ref_si)) - 1.0
            
        if np.any(np.isnan(si)):
            si[np.isnan(si)] = 0.

        si_n = si
        return si_n
       
    def to_categorical(self, data, macs, rooms):
        
        nmac = len(macs)
        nrooms = len(rooms)
        nentry = len(data)
        
        X_train = np.zeros((nentry,nmac))
        Y_train = np.zeros((nentry,nrooms))
        
        for i,r in enumerate(data):
            room_ohe = np.zeros(nrooms,dtype='int')
            room_ohe[rooms.index(r['room'])] = 1
            Y_train[i,:] = room_ohe
            si = self.normalised_signal(r['signal'],r['mac'],macs)
            X_train[i,:] = si
            
        return X_train,Y_train
    
    def plot_data(self, data, macs, rooms, sample=None, sample_name=None):  
        nmac = len(macs)
        nrooms = len(rooms)
        
        sig = np.zeros((nrooms,nmac),dtype='float') #- 127
        for i,r in enumerate(rooms):
            ent = [rm for rm in data if rm['room']==r]
            nent = len(ent)
            si = np.zeros((nent,nmac))
        
            for ei,e in enumerate(ent):
                for j,m in enumerate(macs):
                    if m in e['mac']:
                        si[ei,j] = float(e['signal'][e['mac'].index(m)])
                #Normalise
                si[ei,:] = self.normalised_signal(si[ei,:],self.macs,self.macs)
        
            sig_mean = np.mean(si,axis=0)
            #sig_sd = np.std(si,axis=0)
            sig[i,:] = sig_mean
    
        from matplotlib import pyplot as plt
        width = 0.5
        if sample is None:
            nplots = nrooms
        else:
            nplots = nrooms + 1
        fig, axarr = plt.subplots(nplots)
        ind = np.arange(nmac)
        color = ['r','b','y','g','r','b','y','g']
        for i in range(nrooms):
            rects = axarr[i].bar(ind+(i*width),sig[i,:],width,color=color[i])
            axarr[i].set_title(rooms[i])
        if sample is not None:
            i += 1
            rects = axarr[i].bar(ind+(i*width),sample,width,color=color[i])
            if sample_name is None:
                sample_name = 'Sample'
            axarr[i].set_title(sample_name)
        axarr[-1].set_xlabel('MAC index')
        plt.show()
        
    def create_model(self,nrooms=None,nmacs=None):
        
        from keras.models import Sequential, Model
        from keras.optimizers import sgd, Adam
        from keras.layers.merge import Add # Keras 2.x
        from keras.layers import Dense, Flatten, Input, merge, Lambda, Reshape
        from keras.layers.core import Dense, Dropout, Activation, Flatten
        from keras.layers.convolutional import Convolution2D, MaxPooling2D
        from keras.metrics import categorical_accuracy
        from keras.constraints import maxnorm
        import keras.backend as K
        
        inlayer = Input((nmacs,))
        dl = Dense(100, activation='relu')(inlayer)
        dl = Dropout(0.2)(dl)
        dl = Dense(200, activation='relu')(dl)
        dl = Dropout(0.2)(dl)
        output = Dense(nrooms, activation='softmax')(dl)
        model = Model(inputs=inlayer,outputs=output)
        #opt = sgd()#lr=0.001)
        opt = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[categorical_accuracy])
        return model
        
    def load_weights(self):
        self.model.load_weights(self.model_weights_file)
    
    def train(self,save=True,load_weights=False):
        data = self.read_calibration(n_samples_per_room=10000000)

        if load_weights:
            self.load_weights(model)

        nmacs = len(self.macs)
        nrooms = len(self.rooms)
        nentry = len(data)
        
        X_train,Y_train = self.to_categorical(data, macs, rooms)
        
        if self.model is None:
            self.model = self.create_model(nrooms=nrooms,nmacs=nmacs)
        
        self.model.fit(X_train, Y_train, epochs=2000, verbose=1)
        if save:
            self.model.save_weights(self.model_weights_file)

    def predict(self, macs, data, verbose=False, normalise=True):
        if not np.any([m in self.macs for m in macs]):
            return 'Unknown', 0.
        if normalise:
            ndata = self.normalised_signal(data,macs,self.macs)
        else:
            ndata = data
        pred = self.model.predict(np.reshape(ndata,(1,len(self.macs))))
        order = np.argsort(pred)[0][::-1]
        room_pred = self.rooms[order[0]]
        room_prob = pred[0,order[0]]
        if verbose:
            print(' ')
            print('PREDICTION:')
            for i,o in enumerate(order):
                print('{}, {}'.format(self.rooms[o],pred[0,o]))
            print(' ')
            print('Sample room: {}'.format(sampleRoom))
        return room_pred, room_prob
