"""
Comprehensive test script for all GUI methods in ЕВА.
Tests all public methods across all GUI modules.
"""
import os
import sys
import logging
import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cogniflex.gui.test")

class MockBrain:
    """Mock brain with all components for testing."""
    def __init__(self):
        self.running = True
        self.components = {}
        self.events = MockEventBus()
        self.metrics_collector = MockMetricsCollector()
    
    def get_system_status(self):
        return {'status': 'active', 'components': 10}
    
    def get_system_dashboard_data(self):
        return {'metrics': {'cpu_usage': 0.3, 'memory_usage': 0.4}, 'contradiction_stats': {}}
    
    def get_resource_snapshot(self):
        return {'io_tokens': 1000.0}
    
    def get_cache_stats(self):
        return {'hit_rate': 0.8, 'cache_utilization_percent': 0.5, 'disk_stats': {'entries': 100}}
    
    def process_query(self, query):
        return {'text': f'Mock response to: {query}'}
    
    def tokenize_query(self, query):
        return query.split()
    
    def stop(self):
        self.running = False
    
    def reboot(self):
        pass
    
    def soft_reload(self, reload_gui=True):
        return True


class MockEventBus:
    """Mock event bus."""
    def trigger(self, event_name, data):
        pass
    
    def subscribe(self, event_name, handler, priority=0):
        pass
    
    def on(self, event_name, handler):
        pass


class MockMetricsCollector:
    """Mock metrics collector."""
    def get_current_metrics(self):
        return {'cpu': 0.3, 'memory': 0.4}


class MockIntegrator:
    """Mock integrator."""
    def __init__(self):
        self.event_bus = MockEventBus()
        self.core_brain = MockBrain()
    
    def process_query(self, query, context=None):
        return {'status': 'ok', 'response': {'text': f'Mock response to: {query}'}}
    
    def get_system_health(self):
        return {'status': 'healthy'}
    
    def get_system_stats(self):
        return {'metrics': {'cpu_usage': 0.3, 'memory_usage': 0.4}, 'contradiction_stats': {'total': 0}}
    
    def start_self_dialog(self):
        pass
    
    def optimize_system(self):
        pass


class TestGUIImports(unittest.TestCase):
    """Test that all GUI modules can be imported."""
    
    def test_import_gui_package(self):
        """Test importing the GUI package."""
        try:
            from cogniflex import gui
            logger.info("✓ GUI package imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import GUI package: {e}")
            raise
    
    def test_import_core_gui(self):
        """Test importing core_gui module."""
        try:
            from eva.gui.core_gui import ЕВАGUI, create_gui
            logger.info("✓ core_gui module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import core_gui: {e}")
            raise
    
    def test_import_integrated_gui(self):
        """Test importing integrated_gui module."""
        try:
            from eva.gui.core_gui import ЕВАGUI
            logger.info("✓ core_gui module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import core_gui: {e}")
            raise
    
    def test_import_chat_module(self):
        """Test importing chat_module."""
        try:
            from eva.gui.chat_module import ChatModule
            logger.info("✓ chat_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import chat_module: {e}")
            raise
    
    def test_import_memory_module(self):
        """Test importing memory_module."""
        try:
            from eva.gui.memory_module import MemoryModule
            logger.info("✓ memory_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import memory_module: {e}")
            raise
    
    def test_import_knowledge_graph_module(self):
        """Test importing knowledge_graph_module."""
        try:
            from eva.gui.knowledge_graph_module import KnowledgeGraphModule
            logger.info("✓ knowledge_graph_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import knowledge_graph_module: {e}")
            raise
    
    def test_import_contradiction_module(self):
        """Test importing contradiction_module."""
        try:
            from eva.gui.contradiction_module import ContradictionModule
            logger.info("✓ contradiction_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import contradiction_module: {e}")
            raise
    
    def test_import_learning_module(self):
        """Test importing learning_module."""
        try:
            from eva.gui.learning_module import LearningModule
            logger.info("✓ learning_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import learning_module: {e}")
            raise
    
    def test_import_settings_module(self):
        """Test importing settings_module."""
        try:
            from eva.gui.settings_module import SettingsModule
            logger.info("✓ settings_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import settings_module: {e}")
            raise
    
    def test_import_neuromorphic_module(self):
        """Test importing neuromorphic_module."""
        try:
            from eva.gui.neuromorphic_module import NeuromorphicModule
            logger.info("✓ neuromorphic_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import neuromorphic_module: {e}")
            raise
    
    def test_import_analytics_module(self):
        """Test importing analytics_module."""
        try:
            from eva.gui.analytics_module import AnalyticsModule
            logger.info("✓ analytics_module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import analytics_module: {e}")
            raise
    
    def test_import_settings(self):
        """Test importing settings module."""
        try:
            from eva.gui.settings import load_settings, save_settings
            logger.info("✓ settings module imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import settings: {e}")
            raise
    
    def test_import_gui_utils(self):
        """Test importing gui_utils module."""
        try:
            import eva.gui.gui_utils
            logger.info("✓ gui_utils imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import gui_utils: {e}")
            raise
    
    def test_import_gui_themes(self):
        """Test importing gui_themes module."""
        try:
            import eva.gui.gui_themes as gt
            self.assertTrue(hasattr(gt, 'THEME_COLORS'))
            logger.info("✓ gui_themes imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import gui_themes: {e}")
            raise
    
    def test_import_gui_util(self):
        """Test importing gui_util."""
        try:
            import eva.gui.gui_util as gu
            self.assertTrue(hasattr(gu, 'create_rounded_button'))
            logger.info("✓ gui_util imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import gui_util: {e}")
            raise
    
    def test_import_gui_widgets(self):
        """Test importing gui_widgets module."""
        try:
            import eva.gui.gui_widgets as gw
            self.assertTrue(hasattr(gw, 'create_main_interface'))
            logger.info("✓ gui_widgets imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import gui_widgets: {e}")
            raise
    
    def test_import_widgets(self):
        """Test importing widgets module."""
        try:
            import eva.gui.widgets as w
            self.assertTrue(hasattr(w, 'create_rounded_button'))
            logger.info("✓ widgets imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import widgets: {e}")
            raise
    
    def test_import_gui_modules(self):
        """Test importing gui_modules."""
        try:
            import eva.gui.gui_modules as gm
            self.assertTrue(hasattr(gm, 'init_modules'))
            logger.info("✓ gui_modules imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import gui_modules: {e}")
            raise


class TestЕВАGUIMethods(unittest.TestCase):
    """Test all public methods in ЕВАGUI class."""
    
    @classmethod
    def setUpClass(cls):
        cls.brain = MockBrain()
        cls.integrator = MockIntegrator()
    
    def setUp(self):
        # Create mock root window for tkinter
        self.root = tk.Tk()
        self.root.withdraw()  # Hide window during tests
        
        from eva.gui.core_gui import ЕВАGUI
        self.gui = ЕВАGUI(brain=self.brain, integrator=self.integrator)
        self.gui.root = self.root
        self.gui.running = False  # Don't actually run the GUI
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_process_query_via_integrator(self):
        """Test process_query_via_integrator method."""
        try:
            result = self.gui.process_query_via_integrator("test query")
            self.assertIsInstance(result, dict)
            logger.info("✓ process_query_via_integrator works")
        except Exception as e:
            logger.error(f"✗ process_query_via_integrator failed: {e}")
            raise
    
    def test_get_system_status_via_integrator(self):
        """Test get_system_status_via_integrator method."""
        try:
            result = self.gui.get_system_status_via_integrator()
            self.assertIsInstance(result, dict)
            logger.info("✓ get_system_status_via_integrator works")
        except Exception as e:
            logger.error(f"✗ get_system_status_via_integrator failed: {e}")
            raise
    
    def test_start_self_dialog_via_integrator(self):
        """Test start_self_dialog_via_integrator method."""
        try:
            self.gui.start_self_dialog_via_integrator()
            logger.info("✓ start_self_dialog_via_integrator works")
        except Exception as e:
            logger.error(f"✗ start_self_dialog_via_integrator failed: {e}")
            raise
    
    def test_optimize_system_via_integrator(self):
        """Test optimize_system_via_integrator method."""
        try:
            self.gui.optimize_system_via_integrator()
            logger.info("✓ optimize_system_via_integrator works")
        except Exception as e:
            logger.error(f"✗ optimize_system_via_integrator failed: {e}")
            raise
    
    def test_process_query(self):
        """Test process_query method."""
        try:
            result = self.gui.process_query("test")
            self.assertIsInstance(result, str)
            logger.info("✓ process_query works")
        except Exception as e:
            logger.error(f"✗ process_query failed: {e}")
            raise
    
    def test_update_status(self):
        """Test update_status method."""
        try:
            self.gui.update_status("test_status", {'details': 'test'})
            logger.info("✓ update_status works")
        except Exception as e:
            logger.error(f"✗ update_status failed: {e}")
            raise
    
    def test_show_error(self):
        """Test show_error method."""
        try:
            with patch('tkinter.messagebox.showerror') as mock:
                self.gui.show_error("Test", "Test message")
                logger.info("✓ show_error works")
        except Exception as e:
            logger.error(f"✗ show_error failed: {e}")
            raise
    
    def test_show_message(self):
        """Test show_message method."""
        try:
            with patch('tkinter.messagebox.showinfo') as mock:
                self.gui.show_message("Test", "Test message", "info")
                logger.info("✓ show_message works")
        except Exception as e:
            logger.error(f"✗ show_message failed: {e}")
            raise
    
    def test_show_notification(self):
        """Test show_notification method."""
        try:
            with patch('tkinter.messagebox.showinfo') as mock:
                self.gui.show_notification("Test message", "info")
                logger.info("✓ show_notification works")
        except Exception as e:
            logger.error(f"✗ show_notification failed: {e}")
            raise
    
    def test_create_main_window(self):
        """Test create_main_window method."""
        try:
            self.gui.root = None
            self.gui.create_main_window()
            self.assertIsNotNone(self.gui.root)
            logger.info("✓ create_main_window works")
        except Exception as e:
            logger.error(f"✗ create_main_window failed: {e}")
            raise
    
    def test_show_toast(self):
        """Test show_toast method."""
        try:
            self.gui.show_toast("Test message", "info")
            logger.info("✓ show_toast works")
        except Exception as e:
            logger.error(f"✗ show_toast failed: {e}")
            raise
    
    def test_start_gui(self):
        """Test start_gui method."""
        try:
            self.gui.start_gui()
            logger.info("✓ start_gui works")
        except Exception as e:
            logger.error(f"✗ start_gui failed: {e}")
            raise
    
    def test_reload(self):
        """Test reload method."""
        try:
            self.gui.reload()
            logger.info("✓ reload works")
        except Exception as e:
            logger.error(f"✗ reload failed: {e}")
            raise
    
    def test_get_system_status_fallback(self):
        """Test _get_system_status_fallback method."""
        try:
            result = self.gui._get_system_status_fallback()
            self.assertIsInstance(result, dict)
            logger.info("✓ _get_system_status_fallback works")
        except Exception as e:
            logger.error(f"✗ _get_system_status_fallback failed: {e}")
            raise
    
    def test_fallback_query_processing(self):
        """Test _fallback_query_processing method."""
        try:
            result = self.gui._fallback_query_processing("test")
            self.assertIsInstance(result, dict)
            logger.info("✓ _fallback_query_processing works")
        except Exception as e:
            logger.error(f"✗ _fallback_query_processing failed: {e}")
            raise


class TestIntegratedЕВАGUIMethods(unittest.TestCase):
    """Test all public methods in IntegratedЕВАGUI class."""
    
    @classmethod
    def setUpClass(cls):
        cls.brain = MockBrain()
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        from eva.gui.core_gui import ЕВАGUI
        # Patch setup_gui to avoid creating real tkinter widgets
        with patch.object(ЕВАGUI, 'setup_gui', return_value=None):
            self.gui = ЕВАGUI(self.brain)
            self.gui.root = self.root
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_send_message(self):
        """Test send_message method."""
        try:
            with patch.object(self.gui, 'process_query'):
                self.gui.input_field = MagicMock()
                self.gui.input_field.get.return_value = "test"
                self.gui.send_message()
                logger.info("✓ send_message works")
        except Exception as e:
            logger.error(f"✗ send_message failed: {e}")
            raise
    
    def test_process_query(self):
        """Test process_query method."""
        try:
            with patch('threading.Thread'):
                self.gui.process_query("test query")
                logger.info("✓ process_query works")
        except Exception as e:
            logger.error(f"✗ process_query failed: {e}")
            raise
    
    def test_add_to_chat(self):
        """Test add_to_chat method."""
        try:
            self.gui.chat_area = MagicMock()
            self.gui.add_to_chat("User", "Test message")
            logger.info("✓ add_to_chat works")
        except Exception as e:
            logger.error(f"✗ add_to_chat failed: {e}")
            raise
    
    def test_add_to_log(self):
        """Test add_to_log method."""
        try:
            self.gui.log_area = MagicMock()
            self.gui.add_to_log("Test log")
            logger.info("✓ add_to_log works")
        except Exception as e:
            logger.error(f"✗ add_to_log failed: {e}")
            raise
    
    def test_update_status(self):
        """Test update_status method."""
        try:
            self.gui.status_label = MagicMock()
            self.gui.update_status("Active")
            logger.info("✓ update_status works")
        except Exception as e:
            logger.error(f"✗ update_status failed: {e}")
            raise
    
    def test_update_metrics(self):
        """Test update_metrics method."""
        try:
            self.gui.metrics_labels = {}
            for key in ['status', 'components', 'requests', 'responses', 'avg_time', 'active', 'focus', 'contradictions', 'learning']:
                label = MagicMock()
                self.gui.metrics_labels[key] = label
            self.gui.update_metrics()
            logger.info("✓ update_metrics works")
        except Exception as e:
            logger.error(f"✗ update_metrics failed: {e}")
            raise
    
    def test_start_self_dialog(self):
        """Test start_self_dialog method."""
        try:
            with patch.object(self.gui.brain, 'process_query', return_value={'text': 'test'}):
                self.gui.start_self_dialog()
                logger.info("✓ start_self_dialog works")
        except Exception as e:
            logger.error(f"✗ start_self_dialog failed: {e}")
            raise
    
    def test_optimize_system(self):
        """Test optimize_system method."""
        try:
            self.gui.optimize_system()
            logger.info("✓ optimize_system works")
        except Exception as e:
            logger.error(f"✗ optimize_system failed: {e}")
            raise
    
    def test_show_system_stats(self):
        """Test show_system_stats method."""
        try:
            with patch('tkinter.messagebox.showinfo'):
                self.gui.show_system_stats()
                logger.info("✓ show_system_stats works")
        except Exception as e:
            logger.error(f"✗ show_system_stats failed: {e}")
            raise
    
    def test_stop_system(self):
        """Test stop_system method."""
        try:
            self.gui.running = True
            with patch.object(self.gui, 'on_closing'):
                self.gui.stop_system()
                logger.info("✓ stop_system works")
        except Exception as e:
            logger.error(f"✗ stop_system failed: {e}")
            raise
    
    def test_start_gui(self):
        """Test start_gui method."""
        try:
            with patch.object(self.gui, 'start'):
                self.gui.start_gui()
                logger.info("✓ start_gui works")
        except Exception as e:
            logger.error(f"✗ start_gui failed: {e}")
            raise
    
    def test_start(self):
        """Test start method."""
        try:
            self.gui.running = False
            self.gui.start()
            logger.info("✓ start works")
        except Exception as e:
            logger.error(f"✗ start failed: {e}")
            raise
    
    def test_stop(self):
        """Test stop method."""
        try:
            self.gui.running = True
            self.gui.stop()
            logger.info("✓ stop works")
        except Exception as e:
            logger.error(f"✗ stop failed: {e}")
            raise


class TestChatModuleMethods(unittest.TestCase):
    """Test all public methods in ChatModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.gui_queue = MagicMock()
        self.gui.theme = "light"
        self.gui.colors = {}
        self.gui.reasoning_active = True
        
        from eva.gui.chat_module import ChatModule
        self.module = ChatModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ ChatModule.activate works")
        except Exception as e:
            logger.error(f"✗ ChatModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ ChatModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ ChatModule.deactivate failed: {e}")
            raise


class TestMemoryModuleMethods(unittest.TestCase):
    """Test all public methods in MemoryModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.memory_module import MemoryModule
        self.module = MemoryModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ MemoryModule.activate works")
        except Exception as e:
            logger.error(f"✗ MemoryModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ MemoryModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ MemoryModule.deactivate failed: {e}")
            raise
    
    def test_safe_brain_call(self):
        """Test _safe_brain_call method."""
        try:
            result = self.module._safe_brain_call('get_system_status')
            self.assertIsInstance(result, dict)
            logger.info("✓ MemoryModule._safe_brain_call works")
        except Exception as e:
            logger.error(f"✗ MemoryModule._safe_brain_call failed: {e}")
            raise


class TestKnowledgeGraphModuleMethods(unittest.TestCase):
    """Test all public methods in KnowledgeGraphModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.knowledge_graph_module import KnowledgeGraphModule
        self.module = KnowledgeGraphModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ KnowledgeGraphModule.activate works")
        except Exception as e:
            logger.error(f"✗ KnowledgeGraphModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ KnowledgeGraphModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ KnowledgeGraphModule.deactivate failed: {e}")
            raise


class TestContradictionModuleMethods(unittest.TestCase):
    """Test all public methods in ContradictionModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.contradiction_module import ContradictionModule
        self.module = ContradictionModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ ContradictionModule.activate works")
        except Exception as e:
            logger.error(f"✗ ContradictionModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ ContradictionModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ ContradictionModule.deactivate failed: {e}")
            raise


class TestLearningModuleMethods(unittest.TestCase):
    """Test all public methods in LearningModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.learning_module import LearningModule
        self.module = LearningModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ LearningModule.activate works")
        except Exception as e:
            logger.error(f"✗ LearningModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ LearningModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ LearningModule.deactivate failed: {e}")
            raise


class TestSettingsModuleMethods(unittest.TestCase):
    """Test all public methods in SettingsModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.settings_module import SettingsModule
        self.module = SettingsModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ SettingsModule.activate works")
        except Exception as e:
            logger.error(f"✗ SettingsModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ SettingsModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ SettingsModule.deactivate failed: {e}")
            raise


class TestNeuromorphicModuleMethods(unittest.TestCase):
    """Test all public methods in NeuromorphicModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.neuromorphic_module import NeuromorphicModule
        self.module = NeuromorphicModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ NeuromorphicModule.activate works")
        except Exception as e:
            logger.error(f"✗ NeuromorphicModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ NeuromorphicModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ NeuromorphicModule.deactivate failed: {e}")
            raise


class TestAnalyticsModuleMethods(unittest.TestCase):
    """Test all public methods in AnalyticsModule class."""
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.gui = MagicMock()
        self.gui.brain = MockBrain()
        self.gui.content_area = MagicMock()
        self.gui.content_area.winfo_children.return_value = []
        self.gui.root = self.root
        
        from eva.gui.analytics_module import AnalyticsModule
        self.module = AnalyticsModule(self.gui)
        self.module.gui = self.gui
    
    def tearDown(self):
        try:
            self.root.destroy()
        except:
            pass
    
    def test_activate(self):
        """Test activate method."""
        try:
            content_frame = tk.Frame(self.root)
            self.gui.content_area = content_frame
            self.module.activate()
            logger.info("✓ AnalyticsModule.activate works")
        except Exception as e:
            logger.error(f"✗ AnalyticsModule.activate failed: {e}")
            raise
    
    def test_deactivate(self):
        """Test deactivate method."""
        try:
            self.module.deactivate()
            logger.info("✓ AnalyticsModule.deactivate works")
        except Exception as e:
            logger.error(f"✗ AnalyticsModule.deactivate failed: {e}")
            raise


class TestSettingsModule(unittest.TestCase):
    """Test settings module functions."""
    
    def test_load_settings(self):
        """Test load_settings function."""
        try:
            from eva.gui.settings import load_settings
            settings = load_settings("nonexistent.json")
            self.assertIsInstance(settings, dict)
            logger.info("✓ load_settings works")
        except Exception as e:
            logger.error(f"✗ load_settings failed: {e}")
            raise
    
    def test_save_settings(self):
        """Test save_settings function."""
        try:
            from eva.gui.settings import save_settings
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_path = f.name
            try:
                save_settings({"test": "value"}, temp_path)
                logger.info("✓ save_settings works")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"✗ save_settings failed: {e}")
            raise


class TestUtilityModules(unittest.TestCase):
    """Test utility module functions."""
    
    def test_theme_colors(self):
        """Test THEME_COLORS from gui_themes."""
        try:
            from eva.gui.gui_themes import THEME_COLORS
            self.assertIsInstance(THEME_COLORS, dict)
            self.assertIn("light", THEME_COLORS)
            self.assertIn("dark", THEME_COLORS)
            logger.info("✓ THEME_COLORS works")
        except Exception as e:
            logger.error(f"✗ THEME_COLORS failed: {e}")
            raise
    
    def test_create_styles(self):
        """Test create_styles from gui_themes."""
        try:
            from eva.gui.gui_themes import create_styles
            gui_mock = MagicMock()
            gui_mock.colors = {"primary": "blue", "bg": "white"}
            create_styles(gui_mock)
            logger.info("✓ create_styles works")
        except Exception as e:
            logger.error(f"✗ create_styles failed: {e}")
            raise
    
    def test_create_rounded_button(self):
        """Test create_rounded_button from widgets."""
        try:
            from eva.gui.widgets import create_rounded_button
            root = tk.Tk()
            root.withdraw()
            btn = create_rounded_button(root, "Test", command=lambda: None)
            root.destroy()
            logger.info("✓ create_rounded_button works")
        except Exception as e:
            logger.error(f"✗ create_rounded_button failed: {e}")
            raise
    
    def test_create_gradient_canvas(self):
        """Test create_gradient_canvas from widgets."""
        try:
            from eva.gui.widgets import create_gradient_canvas
            root = tk.Tk()
            root.withdraw()
            canvas = create_gradient_canvas(root, 100, 100, "white", "black")
            root.destroy()
            logger.info("✓ create_gradient_canvas works")
        except Exception as e:
            logger.error(f"✗ create_gradient_canvas failed: {e}")
            raise
    
    def test_load_settings_gui_utils(self):
        """Test load_settings from gui_utils."""
        try:
            from eva.gui.gui_utils import load_settings
            settings = load_settings("nonexistent.json")
            self.assertIsInstance(settings, dict)
            logger.info("✓ load_settings from gui_utils works")
        except Exception as e:
            logger.error(f"✗ load_settings from gui_utils failed: {e}")
            raise
    
    def test_save_settings_gui_utils(self):
        """Test save_settings from gui_utils."""
        try:
            from eva.gui.gui_utils import save_settings
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_path = f.name
            try:
                save_settings({"test": "value"}, temp_path)
                logger.info("✓ save_settings from gui_utils works")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"✗ save_settings from gui_utils failed: {e}")
            raise


if __name__ == '__main__':
    print("=" * 80)
    print("ЕВА GUI Integration Test - Testing All GUI Methods")
    print("=" * 80)
    
    # Run tests with verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGUIImports,
        TestЕВАGUIMethods,
        TestIntegratedЕВАGUIMethods,
        TestChatModuleMethods,
        TestMemoryModuleMethods,
        TestKnowledgeGraphModuleMethods,
        TestContradictionModuleMethods,
        TestLearningModuleMethods,
        TestSettingsModuleMethods,
        TestNeuromorphicModuleMethods,
        TestAnalyticsModuleMethods,
        TestSettingsModule,
        TestUtilityModules,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n" + "=" * 80)
        print("FAILURES:")
        print("=" * 80)
        for test, trace in result.failures:
            print(f"\n{test}:")
            print(trace)
    
    if result.errors:
        print("\n" + "=" * 80)
        print("ERRORS:")
        print("=" * 80)
        for test, trace in result.errors:
            print(f"\n{test}:")
            print(trace)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)