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
                          search_duration_seconds,
                          batch_size
                        ) 
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
                          search['search_duration_seconds'],
                          search['batch_size']
                        )                       
                       )
    conn.close()


def insert_trial(trial, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"""INSERT INTO trials 
                          ( kt_trial_id,
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
                          trial['kt_trial_id'],
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
                            trial_uid = %s 
                        """,
                        ( 
                            trial_updated['average_fraction_of_heart_in_mri_batch'],
                            trial_updated['average_fraction_of_pos_gradients_in_heart_in_batch_of_mris'],
                            trial_updated['average_fraction_of_neg_gradients_in_heart_in_batch_of_mris'],
                            trial_updated['trial_uid']
                        )
                      )
    conn.close()


def get_trial_by_trial_uid(trial_uid, database_connection_details):
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
      cursor.execute("SELECT * FROM trials WHERE trial_uid = %s", (trial_uid,))
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


def get_trial_uid_by_performance_metric(metric, ordering, database_connection_details):
  if metric in Model_Metrics._member_names_ and ordering in Order_By._member_names_ :
    conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
    cursor = conn.cursor()
    with conn:
        cursor.execute(f"SELECT trial_uid FROM trials ORDER BY {metric} {ordering}")
        results = cursor.fetchall()
    conn.close()
    return results
  elif metric not in Model_Metrics._member_names_:
    return 'Error - Requested to order trials by an invalid metric name. Metric param should be in Config.Model_Metrics._member_names_'
  elif ordering not in Order_By._member_names_:
    return 'Error - Requested to order trials by an invalid ordering option. Ordering param be in Config.Order_By._member_names_'  
  else:
    return 'Error - Requested to order trials by an invalid metric name or ordering option. Metric param should be in Config.Model_Metrics._member_names_ . Ordering param be in Config.Order_By._member_names_'


def get_trial_and_search_data_by_trial_uid(trial_uid, database_connection_details):
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
                          batch_size,
                          trial_uid,
                          kt_trial_id,
                          model_path,
                          val_loss,
                          val_acc,
                          train_loss,
                          train_acc,
                          last_conv_layer_name,
                          c1_train_acc,
                          c2_train_acc,
                          c1_val_acc,
                          c2_val_acc,
                          average_fraction_of_heart_in_mri_batch,
                          average_fraction_of_pos_gradients_in_heart_in_batch_of_mris,
                          average_fraction_of_neg_gradients_in_heart_in_batch_of_mris
                        FROM trials
                        INNER JOIN tuner_search ON trials.search_id = tuner_search.search_id
                        WHERE trials.trial_uid = %s""", (trial_uid,)
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
    'search_duration_seconds' : results[16],
    'batch_size' : results[17]
  }
  trial = {
    'trial_uid' : results[18], 
    'kt_trial_id' : results[19], 
    'search_id' : results[0], 
    'model_path' : results[20], 
    'val_loss' : results[21], 
    'val_acc' : results[22], 
    'train_loss' : results[23], 
    'train_acc' : results[24], 
    'last_conv_layer_name' : results[25], 
    'c1_train_acc' : results[26],
    'c2_train_acc' : results[27],
    'c1_val_acc' : results[28],
    'c2_val_acc' : results[29],
    'average_fraction_of_heart_in_mri_batch' : results[30], 
    'average_fraction_of_pos_gradients_in_heart_in_batch_of_mris' : results[31], 
    'average_fraction_of_neg_gradients_in_heart_in_batch_of_mris' : results[32]
  }

  return trial, tuner_search


def get_trial_uids_without_grad_cam_overlap_metrics(database_connection_details):
  conn = psycopg2.connect(database="postgres", user = database_connection_details['user'], host = database_connection_details['ngrok_host'] , port = database_connection_details['ngrok_port'])
  cursor = conn.cursor()
  with conn:
      cursor.execute(f""" SELECT trial_uid FROM trials 
                          WHERE 
                                  average_fraction_of_heart_in_mri_batch                        IS NULL 
                              AND average_fraction_of_pos_gradients_in_heart_in_batch_of_mris   IS NULL 
                              AND average_fraction_of_neg_gradients_in_heart_in_batch_of_mris   IS NULL
                      """
                    )
      results = cursor.fetchall()
  conn.close()
  return results

