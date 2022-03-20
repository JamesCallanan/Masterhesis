import psycopg2
from config import Model_Metrics, Order_By

def insert_tuner_search(search, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"""INSERT INTO tuner_search 
                        ( search_id ,
                          search_type,
                          num_models,
                          num_epochs,
                          model_template_builder_name,
                          hyperparam_ranges,
                          disease_classes,
                          model_mode,
                          perform_ROI,
                          depth,
                          width,
                          height,
                          git_commit_id,
                          git_branch,
                          tensorboard_folder_path,
                          keras_tuner_folder_path,
                          search_duration_seconds
                        ) 
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING
                        """, 
                        ( 
                          search['search_id'],
                          search['search_type'],
                          search['num_models'],
                          search['num_epochs'],
                          search['model_template_builder_name'],
                          search['hyperparam_ranges'],
                          search['disease_classes'],
                          search['model_mode'],
                          search['perform_ROI'],
                          search['depth'],
                          search['width'],
                          search['height'],
                          search['git_commit_id'],
                          search['git_branch'],
                          search['tensorboard_folder_path'],
                          search['keras_tuner_folder_path'],
                          search['search_duration_seconds']
                        )                       
                       )
    conn.close()


def insert_trial(trial, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"""INSERT INTO trials 
                          ( trial_id,
                            search_id,
                            model_path,
                            val_loss,
                            val_acc,
                            train_loss,
                            train_acc,
                            last_conv_layer_name
                          )
                          VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                          ON CONFLICT DO NOTHING
                        """,
                        (
                          trial['trial_id'],
                          trial['search_id'],
                          trial['model_path'],
                          trial['val_loss'],
                          trial['val_acc'],
                          trial['train_loss'],
                          trial['train_acc'],
                          trial['last_conv_layer_name']
                        )
                      )
    conn.close()


def update_trial_with_heatmap_data(trial_updated, database_connection_details): 
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute( """ UPDATE trials 
                            SET 
                              average_fraction_of_heart_in_mri_batch = %s,
                              average_fraction_of_pos_gradients_in_heart_in_batch_of_mris = %s,
                              average_fraction_of_neg_gradients_in_heart_in_batch_of_mris = %s
                            WHERE 
                            trial_id = %s 
                        """,
                        ( 
                            trial_updated['average_fraction_of_heart_in_mri_batch'],
                            trial_updated['average_fraction_of_pos_gradients_in_heart_in_batch_of_mris'],
                            trial_updated['average_fraction_of_neg_gradients_in_heart_in_batch_of_mris'],
                            trial_updated['trial_id']
                        )
                      )
    conn.close()


def get_trial_by_id(trial_id, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
      cursor.execute("SELECT * FROM trials WHERE trial_id = %s", (trial_id,))
      results = cursor.fetchall()
    conn.close()
    return results


def get_tuner_search_by_id(search_id, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
      cursor.execute(f"SELECT * FROM tuner_search WHERE search_id = '%s'", (search_id,))
      results = cursor.fetchall()
    conn.close()
    return results

def get_all_trials(database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute("SELECT * FROM trials")
        results = cursor.fetchall()
    conn.close()
    return results

def get_all_tuner_searches(database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute("SELECT * FROM tuner_search")
        results = cursor.fetchall()
    conn.close()
    return results


def get_trial_id_by_performance_metric(metric, ordering, database_connection_details):
  if metric in Model_Metrics._member_names_ and ordering in Order_By._member_names_ :
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"SELECT trial_id FROM trials ORDER BY {metric} {ordering}")
        results = cursor.fetchall()
    conn.close()
    return results
  elif metric not in Model_Metrics._member_names_:
    return 'Error - Requested to order trials by an invalid metric name. Metric param should be in Config.Model_Metrics._member_names_'
  elif ordering not in Order_By._member_names_:
    return 'Error - Requested to order trials by an invalid ordering option. Ordering param be in Config.Order_By._member_names_'  
  else:
    return 'Error - Requested to order trials by an invalid metric name or ordering option. Metric param should be in Config.Model_Metrics._member_names_ . Ordering param be in Config.Order_By._member_names_'


def get_trial_and_search_data_by_trial_id(trial_id, database_connection_details):
  conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
  cursor = conn.cursor()
  with conn:
      cursor.execute("""SELECT
                          trials.search_id,
                          search_type,
                          num_models,
                          num_epochs,
                          model_template_builder_name,
                          hyperparam_ranges,
                          disease_classes,
                          model_mode,
                          perform_ROI,
                          depth,
                          width,
                          height,
                          git_commit_id,
                          git_branch,
                          tensorboard_folder_path,
                          keras_tuner_folder_path,
                          search_duration_seconds,
                          trial_id,
                          model_path,
                          val_loss,
                          val_acc,
                          train_loss,
                          train_acc,
                          last_conv_layer_name,
                          average_fraction_of_heart_in_mri_batch,
                          average_fraction_of_pos_gradients_in_heart_in_batch_of_mris,
                          average_fraction_of_neg_gradients_in_heart_in_batch_of_mris
                        FROM trials
                        INNER JOIN tuner_search ON trials.search_id = tuner_search.search_id
                        WHERE trials.trial_id = %s""", (trial_id,)
      )
      results = cursor.fetchone()
  conn.close()
  tuner_search = {
    'search_id' : results[0], 
    'search_type' : results[1], 
    'num_models' : results[2], 
    'num_epochs' : results[3], 
    'model_template_builder_name' : results[4], 
    'hyperparam_ranges' : results[5], 
    'disease_classes' : results[6], 
    'model_mode' : results[7], 
    'perform_ROI' : results[8], 
    'depth' : results[9], 
    'width' : results[10], 
    'height' : results[11], 
    'git_commit_id' : results[12], 
    'git_branch' : results[13], 
    'tensorboard_folder_path' : results[14], 
    'keras_tuner_folder_path' : results[15],
    'search_duration_seconds' : results[16]
  }
  trial = {
    'trial_id' : results[17], 
    'search_id' : results[0], 
    'model_path' : results[18], 
    'val_loss' : results[19], 
    'val_acc' : results[20], 
    'train_loss' : results[21], 
    'train_acc' : results[22], 
    'last_conv_layer_name' : results[23], 
    'average_fraction_of_heart_in_mri_batch' : results[24], 
    'average_fraction_of_pos_gradients_in_heart_in_batch_of_mris' : results[25], 
    'average_fraction_of_neg_gradients_in_heart_in_batch_of_mris' : results[26]
  }

  return trial, tuner_search
