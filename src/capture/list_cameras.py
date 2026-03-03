import cv2

def list_ports():
    print("Scanning for available camera ports (this may take a few seconds)...")
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    
    while is_working and dev_port < 10:
        # Check standard CAP_ANY
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            # Check other backends just in case DSHOW is needed on Windows
            camera = cv2.VideoCapture(dev_port, cv2.CAP_DSHOW)
            if camera.isOpened():
                 is_working = True
                 print(f"Port {dev_port} is working (via DSHOW backend).")
                 working_ports.append(dev_port)
            else:
                 print(f"Port {dev_port} is not responding.")
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f"Port {dev_port} is working and reading images ({w} x {h}).")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port} for camera ( {w} x {h}) is present but does not read.")
                available_ports.append(dev_port)
        camera.release()
        
        # We only really care about 0,1,2,3 for most people, 
        # but let's hard cap at 5 to save time unless requested
        if dev_port == 4: 
             is_working = False
        dev_port += 1
        
        
    print(f"\nSummary:")
    print(f"Fully Working Ports: {working_ports}")
    print(f"Detected but not reading: {available_ports}")

if __name__ == '__main__':
    list_ports()
