import logging
import os
from datetime import datetime

class SQLLogger:
    def __init__(self):
        self.logger = logging.getLogger('sql_flow')
        self.logger.setLevel(logging.DEBUG)
        self.current_query_status = {
            'initial_success': False,
            'required_recovery': False,
            'final_success': False
        }
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'logs/sql_flow_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_step(self, step_name: str, **kwargs):
        """Log a step in the SQL generation/correction process"""
        self.logger.info(f"\n{'='*50}\n{step_name}\n{'='*50}")
        
        # Update status based on step name
        if step_name == "Initial SQL Generation":
            self.current_query_status['initial_success'] = True
        elif step_name == "SQL Error Recovery":
            self.current_query_status['required_recovery'] = True
            self.current_query_status['initial_success'] = False
        elif step_name == "Final SQL Correction":
            self.current_query_status['required_recovery'] = True
            self.current_query_status['initial_success'] = False
            
        # Add status to log if this is a query execution
        if 'sql_query' in kwargs:
            status_msg = "\nQuery Status:"
            if self.current_query_status['initial_success']:
                status_msg += "\n- Query succeeded on first attempt"
            if self.current_query_status['required_recovery']:
                status_msg += "\n- Query required error recovery"
            self.logger.info(status_msg)
            
        for key, value in kwargs.items():
            self.logger.info(f"\n{key}:\n{value}")
            
    def log_error(self, step_name: str, error: str):
        """Log an error in the process"""
        self.logger.error(f"\n{'='*50}\nERROR in {step_name}\n{'='*50}\n{error}")
        self.current_query_status['initial_success'] = False
        self.current_query_status['required_recovery'] = True
        
    def reset_status(self):
        """Reset the status tracking for a new query"""
        self.current_query_status = {
            'initial_success': False,
            'required_recovery': False,
            'final_success': False
        }

# Global logger instance
sql_logger = SQLLogger()
