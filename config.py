# input params
input_dim = 4
output_dim = 4
hidden_dim = 64
n_class = 5
input_timestep = 7
output_timestep = 7
noise_dim = 4
batch_size = 512
lr=1e-4
d_iter=5
epochs=500
devices=['e', '1', '14', '20', '27', '30']
attributes=['PM2_5', 'PM10', 'humidity', 'temp']
warm_start_path = 'fimi/best_gen.torch'
