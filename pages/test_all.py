import pytest
from back import load_model



def test_load_model_file_not_found():
    # Appeler la fonction load_model avec un numéro de cluster inexistant
    model = load_model(999)

    # Vérifier que la fonction renvoie None lorsque le fichier n'est pas trouvé
    assert model is None



import pytest
from unittest.mock import patch
from back import fetch_data_from_supabase

@pytest.fixture
def mock_supabase():
    with patch('back.supabase') as mock_supabase:
        yield mock_supabase


def test_fetch_data_from_supabase_no_data(mock_supabase):
    # Configuration du mock pour simuler une réponse sans données
    mock_response = {'data': []}
    mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
    
    # Appel de la fonction
    result = fetch_data_from_supabase('test_table')
    
    # Vérification du résultat
    assert result is None

def test_fetch_data_from_supabase_error(mock_supabase):
    # Configuration du mock pour simuler une erreur
    mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.side_effect = Exception('Test error')
    
    # Appel de la fonction
    result = fetch_data_from_supabase('test_table')
    
    # Vérification du résultat
    assert result is None

from back import retard_facture

def test_retard_facture_en_avance():
    row = {'type_retard': 0}
    assert retard_facture(row) == 'En avance'

def test_retard_facture_a_temps():
    row = {'type_retard': 1}
    assert retard_facture(row) == 'à temps'

def test_retard_facture_en_retard():
    row = {'type_retard': 2}
    assert retard_facture(row) == 'En retard'

def test_retard_facture_valeur_manquante():
    row = {'type_retard': None}
    assert retard_facture(row) == ''