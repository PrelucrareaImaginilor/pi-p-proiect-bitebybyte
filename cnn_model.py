import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import Hyperband, Objective
from tensorflow.keras.regularizers import l2

# Custom RMSE loss function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Load and prepare data
data_path = "D:\\Silviu\\facultate\\an3_proiect_pi\\data_merger\\merged_data2.csv"
data = pd.read_csv(data_path)
print("Data loaded:", data.shape)

# Define feature columns
numerical_features = ['bmi', 'sex', 'race', 'p_factor_fs', 'internalizing_fs', 'externalizing_fs', 'attention_fs']
connectivity_features = [col for col in data.columns if col.startswith('conn_')]
X_fmri = data[connectivity_features].values
X_demographic = data[numerical_features].values
y = data['age'].values

# Normalize demographic data
scaler = StandardScaler()
X_demographic = scaler.fit_transform(X_demographic)

# Split data
X_fmri_train, X_fmri_test, X_demographic_train, X_demographic_test, y_train, y_test = train_test_split(
    X_fmri, X_demographic, y, test_size=0.2, random_state=42
)

# Define model architecture
def build_model(hp):
    fmri_input = Input(shape=(X_fmri_train.shape[1],), name="fmri_input")
    demographic_input = Input(shape=(X_demographic_train.shape[1],), name="demographic_input")

    # fmri branch
    x = Dense(units=hp.Int('units_fmri', 64, 256, step=64), activation='relu')(fmri_input)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    # demographic branch
    y = Dense(units=hp.Int('units_demo', 64, 256, step=64), activation='relu')(demographic_input)
    y = Dropout(0.5)(y)
    y = BatchNormalization()(y)

    # Combine and output
    combined = concatenate([x, y])
    outputs = Dense(1, activation='linear')(combined)
    model = Model(inputs=[fmri_input, demographic_input], outputs=outputs)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), loss=rmse, metrics=[rmse])

    return model

# Set up Hyperband tuner
tuner = Hyperband(
    build_model,
    objective=Objective('val_rmse', direction='min'),
    max_epochs=100,
    factor=3,
    directory='hyperband_tuning',
    project_name='age_prediction'
)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(
    monitor='val_rmse',
    patience=10,
    min_delta=1e-4,  # Small change to recognize as improvement
    mode='min',
    restore_best_weights=True,
    verbose=1  # Add verbose to see logs each epoch
)
reduce_lr = ReduceLROnPlateau(monitor='val_rmse', factor=0.5, patience=5, min_lr=0.0001)

# Run tuning
tuner.search(
    x=[X_fmri_train, X_demographic_train],
    y=y_train,
    epochs=100,
    validation_data=([X_fmri_test, X_demographic_test], y_test),
    callbacks=[early_stopping, reduce_lr]
)

best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_rmse = best_model.evaluate([X_fmri_test, X_demographic_test], y_test)
print(f"Best Model Test RMSE: {test_rmse}")

# Save the best model
best_model.save("best_age_prediction_model_with_rmse_focus.h5")
print("Best model saved as 'best_age_prediction_model_with_rmse_focus.h5'")
