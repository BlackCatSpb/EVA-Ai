"""Launcher for Web GUI ЕВА"""
import sys
import os
import logging
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("eva.webgui")

brain = None

def init_brain():
    """Initialize CoreBrain"""
    global brain
    try:
        logger.info("Initializing CoreBrain...")
        from eva.core.core_brain import CoreBrain
        brain = CoreBrain()
        logger.info("CoreBrain created, initializing...")
        
        if not brain.initialize():
            logger.error("Failed to initialize brain")
            return False
            
        logger.info("Starting brain...")
        if not brain.start():
            logger.error("Failed to start brain")
            return False
            
        logger.info("Brain is ready!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize brain: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    global brain
    
    print("=" * 50)
    print("  ЕВА Web GUI")
    print("  http://127.0.0.1:5555")
    print("  admin / cogniflex")
    print("=" * 50)
    
    # Initialize brain (optional - works in demo mode if fails)
    brain_ready = init_brain()
    if brain_ready:
        print("  Brain: CONNECTED")
    else:
        print("  Brain: DEMO MODE")
    print("=" * 50)
    
    # Import server and create app with brain
    web_gui_dir = os.path.join(script_dir, 'eva', 'gui', 'web_gui')
    sys.path.insert(0, web_gui_dir)
    
    import server
    logger.info(f"Creating app with brain: {brain}")
    gui = server.create_app(brain=brain)
    logger.info(f"GUI brain reference: {gui.brain}")
    print(f"Server started on http://{gui.host}:{gui.port}")
    
    try:
        while True:
                    time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        if brain:
            try:
                brain.stop()
            except:
                pass
        gui.stop()

if __name__ == '__main__':
    main()
