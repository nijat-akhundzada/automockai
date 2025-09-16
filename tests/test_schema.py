
import unittest
from unittest.mock import MagicMock, patch
from sqlalchemy.engine import Engine
from automockai.schema import SchemaAnalyzer, make_engine

class TestSchemaAnalyzer(unittest.TestCase):

    def setUp(self):
        self.mock_engine = MagicMock(spec=Engine)
        self.mock_engine.dialect.name = 'postgresql'
        self.analyzer = SchemaAnalyzer(self.mock_engine)

    def test_detect_db_type(self):
        self.assertEqual(self.analyzer._detect_db_type(), 'postgresql')
        self.mock_engine.dialect.name = 'mysql'
        self.assertEqual(self.analyzer._detect_db_type(), 'mysql')
        self.mock_engine.dialect.name = 'sqlite'
        self.assertEqual(self.analyzer._detect_db_type(), 'sqlite')
        self.mock_engine.dialect.name = 'unknown'
        self.assertEqual(self.analyzer._detect_db_type(), 'unknown')

    @patch('sqlalchemy.inspect')
    def test_get_table_names(self, mock_inspect):
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = ['table1', 'table2']
        mock_inspect.return_value = mock_inspector
        analyzer = SchemaAnalyzer(self.mock_engine)
        self.assertEqual(analyzer.get_table_names(), ['table1', 'table2'])

    @patch('sqlalchemy.inspect')
    def test_get_column_info(self, mock_inspect):
        mock_inspector = MagicMock()
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'default': None, 'autoincrement': True, 'comment': None},
            {'name': 'name', 'type': 'VARCHAR(50)', 'nullable': False, 'default': None, 'autoincrement': False, 'comment': 'The name'},
        ]
        mock_inspect.return_value = mock_inspector
        analyzer = SchemaAnalyzer(self.mock_engine)
        columns = analyzer.get_column_info('table1')
        self.assertEqual(len(columns), 2)
        self.assertEqual(columns[0]['name'], 'id')
        self.assertEqual(columns[1]['name'], 'name')

if __name__ == '__main__':
    unittest.main()
