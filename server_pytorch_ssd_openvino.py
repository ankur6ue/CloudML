import sys
sys.path.append('/opt/intel/computer_vision_sdk/python/python3.5')
import os
import argparse
import cv2
import time
import numpy as np
import psutil
from flask import Flask, request, Response, jsonify
from openvino.inference_engine import IENetwork, IEPlugin
from threading import Thread
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
#cors = CORS(app)


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
        
    return response


@app.route('/test')
def index():
    return Response('Pytorch object detection')


@app.route('/local')
def local():
    return Response(open('/home/bitnami/apps/object_detect/local.html').read(), mimetype="text/html")

# for storing global variables initialized during setup call that can be accessed during the detect call.
cache = {}

@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_file = request.files['image']  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)
        
        start = time.time()
        image_file_np = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(image_file_np, cv2.IMREAD_UNCHANGED)
        exec_net = cache['exec_net']
        n = cache['n']
        c = cache['c']
        h = cache['h']
        w = cache['w']
        input_blob = cache['input_blob']
        out_blob = cache['out_blob']
        # We expect the client to have resized the image to the proper dimensions. Check that it is the case, if not 
        # resize to correct size. 
        
        # image = cv2.resize(img, (w, h))
        image = img        
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((n, c, h, w))
        # Start sync inference
        res = exec_net.infer(inputs={input_blob: image})
        end = time.time()
        time_elapsed = end - start
        proc = psutil.Process()
        result = {}
        object_data = {}
        object_data["processing_time"] = time_elapsed
        objects = []
        bbox_data = res[out_blob][0][0]
        
        for obj in bbox_data:
            #app.logger.info('processing objects {}'.format(len(obj)))
            if (len(obj) == 7):
                label = obj[1]
                conf = obj[2]
                pt = obj[3:7]
                object = {}
                if (conf > threshold):
                    object['score'] = float(conf)
                    object['class_name'] = str(label)
                    object['x'] = float(pt[0])
                    object['y'] = float(pt[1])
                    object['width'] = float(pt[2] - pt[0])
                    object['height'] = float(pt[3] - pt[1])
                    objects.append(object)

        object_data["objects"] = objects
        return jsonify(object_data)


    except Exception as e:
        print('POST /image error: %e' % e)
        return e

def setup_openvino():
    app.logger.info('in setup_openvino')
    args = {}
    args['model'] = app.config['model']
    args['device'] = app.config['device']
    args['cpu_extension'] = app.config['cpu_extension']
    args['plugin_dir'] = app.config['plugin_dir']
    
    app.logger.info("Using model: {}".format(args['model']))
    model_xml = args['model']
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    app.logger.info("Initializing plugin for {} device...".format(args['device']))
    
    plugin = IEPlugin(device=args['device'], plugin_dirs=args['plugin_dir'])
    if (plugin != None):
        app.logger.info("plugin successfully initialized")
    else:
        app.logger.info("error initializing plugin")
    
    if args['cpu_extension'] and 'CPU' in args['device']:
        plugin.add_cpu_extension(args['cpu_extension'])
    # Read IR
    print("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=1)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]
    del net
    cache['exec_net'] = exec_net
    cache['n'] = n
    cache['c'] = c
    cache['h'] = h
    cache['w'] = w
    cache['input_blob'] = input_blob
    cache['out_blob'] = out_blob
    cache['plugin'] = plugin

@app.route('/init')
def init():
    handler = RotatingFileHandler('/home/bitnami/apps/object_detect/mesg.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    setup_openvino()
    return Response('done')


if __name__ == '__main__':
    
    # without SSL
    #app.run(host='0.0.0.0', port=5000)

    # with SSL
    app.run(debug=True, host='0.0.0.0', ssl_context=('/opt/bitnami/apache2/conf/server.crt', '/opt/bitnami/apache2/conf/server.key'))
