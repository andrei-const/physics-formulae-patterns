import numpy as np 
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit, minimize
import torch
import matplotlib.colors as mcolors
import signal
from contextlib import contextmanager
import pickle
from sklearn.model_selection import train_test_split
import sys
import warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.evidence import HarmonicEvidence, K_EvidenceNetwork

# --------------------------------------------
# Misc functions
# --------------------------------------------

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """ Check function call does not exceed allotted time
    
    Args:
        :seconds (float): maximum time function can run in seconds
    
    Raises:
        TimeoutException if time exceeds seconds
    """
    
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        warnings.filterwarnings("ignore")
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
        warnings.filterwarnings("default")

# --------------------------------------------
# Zipf and Zipf-Mandelbrot generators
# --------------------------------------------

def generate_from_zipf(N, alpha, Nmax=1000):
    """
    Generate a list of N integers from a Zipf distribution with parameter alpha.
    The distribution is truncated at Nmax.
    """
    
    # Generate a list of integers from 1 to Nmax
    x = np.arange(1, Nmax+1)
    # Compute the probabilities of each integer
    p = 1 / np.power(x, alpha)
    # Normalize the probabilities
    p = p / np.sum(p)
    # Generate a sample of size N from the integers with probabilities p
    samps = np.random.choice(x, N, p=p, replace=True)

    # Use Counter to count the occurrences of each integer
    counter = Counter(samps)
    obs = np.array(list(counter.values()))
    obs = np.sort(obs)[::-1]

    return obs


def generate_from_zipf_mandel(N, alpha, b, Nmax=1000):
    """
    Generate a list of N integers from a Zipf-Mandelbrot distribution with parameters alpha and b.
    The distribution is truncated at Nmax.
    """
    
    # Generate a list of integers from 1 to Nmax
    x = np.arange(1, Nmax+1)
    # Compute the probabilities of each integer
    p = 1 / np.power(x+b, alpha)
    # Normalize the probabilities
    p = p / np.sum(p)
    # Generate a sample of size N from the integers with probabilities p
    samps = np.random.choice(x, N, p=p, replace=True)

    # Use Counter to count the occurrences of each integer
    counter = Counter(samps)
    obs = np.array(list(counter.values()))
    obs = np.sort(obs)[::-1]

    return obs


def generate_from_exponential(N, beta, Nmax=1000):

    # Generate a list of integers from 1 to Nmax
    x = np.arange(1, Nmax+1)
    # Compute the probabilities of each integer
    p = np.exp(- beta * x)
    # Normalize the probabilities
    p = p / np.sum(p)
    # Generate a sample of size N from the integers with probabilities p
    samps = np.random.choice(x, N, p=p, replace=True)

    # Use Counter to count the occurrences of each integer
    counter = Counter(samps)
    obs = np.array(list(counter.values()))
    obs = np.sort(obs)[::-1]

    return obs

# --------------------------------------------
# Fitting functions
# --------------------------------------------

def fit_to_zipf(obs):
    """
    Fit the observed distribution to a Zipf distribution and return the estimated alpha.
    """

    rank = np.arange(1, len(obs)+1)

    if len(obs) == 1:
        popt = [1, 0]
        log_mse = 0
    else:
        def zipf(x, A, alpha):
            y = 1 / np.power(x, alpha) / np.sum(1 / np.power(rank, alpha))
            y /= np.sum(y)
            return A * obs.sum() * y
        
        with suppress_stdout():
            popt, _ = curve_fit(zipf, rank, obs)
        log_mse = np.mean(np.power(np.log(zipf(rank, *popt) + 1) - np.log(obs + 1), 2))

    return list(popt) + [log_mse]

def fit_to_zipf_mandel(obs):
    """
    Fit the observed distribution to a Zipf-Mandelbrot distribution and return the estimated alpha.
    """

    rank = np.arange(1, len(obs)+1)

    if len(obs) == 1:
        popt = [1, 0, 0]
        log_mse = 0
    elif len(obs) == 2:
        res = fit_to_zipf(obs)
        popt = res[:-1] + [0] 
        log_mse = res[-1]
    else:

        def zipf_mandel(x, A, alpha, b):
            y = 1 / np.power(x+b, alpha) / np.sum(1 / np.power(rank+b, alpha))
            y /= np.sum(y)
            return A * obs.sum() * y
        
        with suppress_stdout():
            popt, _ = curve_fit(zipf_mandel, rank, obs)
        log_mse = np.mean(np.power(np.log(zipf_mandel(rank, *popt) + 1) - np.log(obs + 1), 2))

    return list(popt) + [log_mse]


def fit_to_exponential(obs):
    """
    Fit the observed distribution to an exponential distribution and return the estimated beta and gamma.
    """

    rank = np.arange(1, len(obs)+1)

    if len(obs) == 1:
        popt = [1, 0]
        log_mse = 0
    else:
        def exponential(x, A, beta):
            y = np.exp(- beta * x)
            y /= np.sum(y)
            return A * obs.sum() * y
        
        with suppress_stdout():
            popt, _ = curve_fit(exponential, rank, obs)
        log_mse = np.mean(np.power(np.log(exponential(rank, *popt) + 1) - np.log(obs + 1), 2))

    return list(popt) + [log_mse]

# --------------------------------------------
# Simulators
# --------------------------------------------

def create_compressed_obs(obs):
    """
    Create a compressed version of the observed distribution.
    """
    zipf_fit = fit_to_zipf(obs)
    zipf_mandel_fit = fit_to_zipf_mandel(obs)
    exponential_fit = fit_to_exponential(obs)

    return np.array(list(zipf_fit) + list(zipf_mandel_fit) + list(exponential_fit) + [len(obs)])


def zipf_simulator(N, inv_alpha, Nmax=1000):
    """
    Generate a sample of size N from a Zipf distribution with parameter alpha and fit the distribution.
    """

    obs = generate_from_zipf(N, 1 / inv_alpha, Nmax)

    return create_compressed_obs(obs)


def zipf_mandel_simulator(N, inv_alpha, inv_b, Nmax=1000):
    """
    Generate a sample of size N from a Zipf distribution with parameter alpha and fit the distribution.
    """

    obs = generate_from_zipf_mandel(N, 1 / inv_alpha, 1 / inv_b, Nmax)

    return create_compressed_obs(obs)


def exponential_simulator(N, beta, Nmax=1000):
    """
    Generate a sample of size N from an exponential distribution with parameter beta and fit the distribution.
    """

    obs = generate_from_exponential(N, beta, Nmax)

    return create_compressed_obs(obs)

# --------------------------------------------
# Alternative fitting functions
# --------------------------------------------

def poisson_fit(obs):

    rank = np.arange(1, len(obs)+1)

    def poisson(alpha):
        lam = np.sum(obs) / np.power(rank, alpha) / np.sum(1 / np.power(rank, alpha))
        nll = -np.sum(np.log(lam) + obs * np.log(lam))
        return nll
    
    res = minimize(poisson, 1, method='Nelder-Mead')

    return res.x[0]

# --------------------------------------------
# Implicit Likelihood Inference
# --------------------------------------------

# Training scripts to train NPE/NLE with LtU-ILI
def train_estimator(loader, prior, x, theta, name, output_dir, engine='NLE'):

    print('\nStarting engine', engine)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(f'{output_dir}/{name}_{engine}'):
        os.mkdir(f'{output_dir}/{name}_{engine}')

    # instantiate your neural networks to be used as an ensemble
    nets = [
        ili.utils.load_nde_sbi(engine=engine, model='maf', hidden_features=50, num_transforms=5),
        ili.utils.load_nde_sbi(engine=engine, model='mdn', hidden_features=50, num_components=6)
    ]
    # nets = [
    #     ili.utils.load_nde_sbi(engine=engine, model='maf', hidden_features=50, num_transforms=5),
    #     ili.utils.load_nde_sbi(engine=engine, model='maf', hidden_features=25, num_transforms=5),
    #     ili.utils.load_nde_sbi(engine=engine, model='mdn', hidden_features=50, num_components=5),
    #     ili.utils.load_nde_sbi(engine=engine, model='mdn', hidden_features=25, num_components=5)
    # ]

    # define training arguments
    train_args = {
        'training_batch_size': 64,
        'learning_rate': 1e-4
    }

    # initialize the trainer
    runner = InferenceRunner.load(
        backend='sbi',
        engine=engine,
        prior=prior,
        nets=nets,
        device=device,
        embedding_net=None,
        train_args=train_args,
        proposal=None,
        out_dir=None
    )

    # train the model
    posterior_ensemble, summaries = runner(loader=loader)

    # save the model
    with open(f'{output_dir}/{name}_{engine}/posterior_ensemble.pkl', 'wb') as f:
        pickle.dump(posterior_ensemble, f)

    # plot train/validation loss
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    c = list(mcolors.TABLEAU_COLORS)
    for i, m in enumerate(summaries):
        ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
        ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
    ax.set_xlim(0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log probability')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{name}_{engine}/loss.png')

    # Drawing samples from the ensemble posterior
    if name == 'zipf':
        labels = [r'$\alpha^{-1}$']
    elif name == 'zipf_mandel':
        labels = [r'$\alpha^{-1}$', r'$b$']
    elif name == 'exponential':
        labels = [r'$\beta$']
    else:
        raise ValueError('Unknown simulator')
    
    if engine == 'NLE':
        sampling_method = 'vi'
        sample_params={'dist': 'maf', 'n_particles': 32, 'learning_rate': 1e-2}
    elif engine == 'NPE':
        sampling_method = 'direct'
        # sampling_method = 'emcee'
        sample_params = {}
    else:
        raise ValueError('Unknown engine')
    
    if engine == 'NLE':
        plt.close('all')
        return posterior_ensemble, summaries
    
    try:
        with time_limit(120):
            metric = PosteriorCoverage(
                num_samples=500, sample_method=sampling_method, 
                sample_params=sample_params,
                labels=labels,
                plot_list = ["coverage", "histogram", "predictions", "tarp", "logprob"],
                out_dir=None
            )
            fig = metric(
                posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
                x=x, theta=theta
            )
    except TimeoutException:
        print('Timeout, changing to mcmc sampling method')
        metric = PosteriorCoverage(
            num_samples=500, sample_method='emcee', 
            sample_params=sample_params,
            labels=labels,
            plot_list = ["coverage", "histogram", "predictions", "tarp", "logprob"],
            out_dir=None
        )
        fig = metric(
            posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
            x=x, theta=theta
        )
    for f in fig:
        f.tight_layout()

    # Figure names
    fig_names = []
    if "coverage" in metric.plot_list:
        fig_names.append("coverage")
    if "histogram" in metric.plot_list:
        fig_names.append("histogram")
    if "predictions" in metric.plot_list:
        fig_names.append("predictions")
    if "tarp" in metric.plot_list:
        fig_names.append("tarp")

    for f, fname in zip(fig, fig_names):
        f.savefig(f'{output_dir}/{name}_{engine}/{fname}.png')

    print('Number of figs', len(fig), len(fig_names))

    plt.close('all')

    return posterior_ensemble, summaries


def train(N, Nsim, output_dir, to_train=['zipf', 'zipf_mandel', 'exponential'], train_nle=False, test_size=0.2):
    """
    Train NPE and NLE on Zipf, Zipf-Mandelbrot and Exponential distributions.

    Args:
        :N (int): number of words in corpus
        :Nsim (int): number of simulations to generate
        :output_dir (str): output directory
        :to_train (list): list of simulators to train
        :train_nle (bool): if True, train NLE
        :test_size (float): test size for train-test split

    Returns:
        :all_nle (list): list of NLEs
        :all_npe (list): list of NPEs
        :all_loader (list): list of loaders
    """

    alpha_min = 1                           # Minimum value of exponent alpha
    alpha_max = 10                          # Maximum value of exponent alpha
    b_min = 2                             # Minimum value of offset b in Zipf-Mandelbrot
    b_max = 20                              # Maximum value of offset b in Zipf-Mandelbrot
    beta_min = 0.1                          # Minimum value of beta in exponential distribution
    beta_max = 3                            # Maximum value of beta in exponential distribution

    all_nle = []
    all_npe = []
    all_loader = []

    for name, sim in zip(['zipf', 'zipf_mandel', 'exponential'], [zipf_simulator, zipf_mandel_simulator, exponential_simulator]):

        print('\n' + '-'*50)
        print('\nSimulator:', name)
        print('\n' + '-'*50 + '\n')

        if name not in to_train:
            
            print('Loading posterior ensembles from file')
            if train_nle:
                with open(f'{output_dir}/{name}_nle/posterior_ensemble.pkl', 'rb') as f:
                    all_nle.append(pickle.load(f))
            try:
                with open(f'{output_dir}/{name}_npe/posterior_ensemble.pkl', 'rb') as f:
                    all_npe.append(pickle.load(f))
            except FileNotFoundError:
                print('NPE not found')
                all_npe.append(None)


        # Generate data
        seed_sim = 12345
        np.random.seed(seed_sim)
        if name == 'zipf':
            theta = np.random.uniform(low=1/alpha_max, high=1/alpha_min, size=Nsim)
            theta = np.expand_dims(theta, axis=1)
        elif name == 'zipf_mandel':
            theta = np.vstack(
                [np.random.uniform(low=1 / alpha_max, high=1 / alpha_min, size=Nsim), 
                np.random.uniform(low=1/b_max, high=1/b_min, size=Nsim)]).T
        elif name == 'exponential':
            theta = np.random.uniform(low=beta_min, high=beta_max, size=Nsim)
            theta = np.expand_dims(theta, axis=1)
        else:
            raise ValueError('Unknown simulator')
        
        x = np.array([sim(N, *t) for t in theta])

        # Test-train split
        x_train, x_test, theta_train, theta_test = train_test_split(x, theta, test_size=test_size, random_state=42)

        print('Theta shape', theta.shape)
        print('X shape', x.shape)

        # make a dataloader
        loader = NumpyLoader(x=x_train, theta=theta_train)
        all_loader.append(loader)

        if name in to_train:

            # define a prior
            if name == 'zipf':
                prior = ili.utils.Uniform(low=[1 / alpha_max], high=[1 / alpha_min], device=device)
            elif name == 'zipf_mandel':
                prior = ili.utils.Uniform(low=[1 / alpha_max, 1 / b_max], high=[1 / alpha_min, 1 / b_min], device=device)
            elif name == 'exponential':
                prior = ili.utils.Uniform(low=[beta_min], high=[beta_max], device=device)
            else:
                raise ValueError('Unknown simulator')

            if train_nle:
                posterior_ensemble, _ = train_estimator(loader, prior, x_test, theta_test, name, output_dir, engine='NLE')
                all_nle.append(posterior_ensemble)
            posterior_ensemble, _ = train_estimator(loader, prior, x_test, theta_test, name, output_dir, engine='NPE')
            all_npe.append(posterior_ensemble)
        else:
            print('Loading posterior ensembles from file')
            if train_nle:
                with open(f'{output_dir}/{name}_nle/posterior_ensemble.pkl', 'rb') as f:
                    all_nle.append(pickle.load(f))
            try:
                with open(f'{output_dir}/{name}_npe/posterior_ensemble.pkl', 'rb') as f:
                    all_npe.append(pickle.load(f))
            except FileNotFoundError:
                print('NPE not found')
                all_npe.append(None)

    return all_nle, all_npe, all_loader


def train_evidence(N, all_nle, all_npe, all_loader, output_dir, evidence_estimator='evidence_network'):
    """
    Train evidence network or harmonic mean estimator.

    Args:
        :N (int): number of words in corpus
        :all_nle (list): list of NLEs
        :all_npe (list): list of NPEs
        :all_loader (list): list of loaders
        :output_dir (str): output directory
        :evidence_estimator (str): 'harmonic' or 'evidence_network'

    """
        
    alpha = 3
    obs = generate_from_zipf(N, alpha)
    obs = create_compressed_obs(obs)

    if evidence_estimator == 'evidence_network':

        print('\nTraining evidence network')

        all_name = ['zipf', 'zipf_mandel', 'exponential']

        for i in range(len(all_loader)-1):

            runner = K_EvidenceNetwork(
                layer_width=64, added_layers=2,
                batch_norm_flag=1, alpha=2,
                train_args=dict(
                    max_epochs=100, lr=1e-9, 
                    training_batch_size=512,
                    stop_after_epochs=100,
                    validation_fraction=0.1)
            )
            
            summary = runner.train(all_loader[-1], all_loader[i])

            with open(f'{output_dir}/{all_name[i]}_{all_name[-1]}_evidence_network.pkl', 'wb') as f:
                pickle.dump(runner, f)

            plt.figure()
            plt.plot(summary['training_loss'], label='Train Loss')
            plt.plot(summary['validation_loss'], label='Val Loss')
            plt.yscale('log')
            plt.ylim(None, min(1, plt.gca().get_ylim()[1] * 2))
            plt.legend()
            plt.savefig(f'{output_dir}/{all_name[i]}_{all_name[-1]}_evidence_loss.png', bbox_inches='tight')
            plt.show()

            if i < len(all_name):
                print(f"Exponential vs {all_name[i]}: Predicted Bayes Factor (positive prefers Exponetial):")
            else:
                print("Unknown comparison")
            for _ in range(20):
                obs = generate_from_zipf(N, alpha)
                obs = create_compressed_obs(obs)
                logK = runner.predict(obs).detach().numpy()[0]
                print(f'Predicted lnK: {-logK[0]}')

    elif evidence_estimator == 'harmonic':

        for i in range(len(all_nle)):
            print(f'\ngetting samples {i}')
            samples = all_npe[i].sample((10_000,), obs, show_progress_bars=True)
            print(samples.min(), samples.max())
            print('getting log probs')
            lnprob = all_nle[i].potential(samples, obs)
            print(lnprob.min(), lnprob.max())
            print(np.unique(lnprob).shape)
            print(obs.shape)
            if i == -1:
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                nplot = 100
                axs[0].hist(samples[:nplot], bins=50, density=True, alpha=0.5)
                axs[1].plot(samples[:nplot])
                plt.show()
            print("SHAPES", samples.shape, lnprob.shape)

        estimator_zipf = HarmonicEvidence()
        estimator_zipf.from_nde(
            all_npe[0], all_nle[0], x=obs,
            shape=(10_000,),
            show_progress_bars=True
        )

        estimator_zipf_mandel = HarmonicEvidence()
        estimator_zipf_mandel.from_nde(
            all_npe[1], all_nle[1], x=obs,
            shape=(10_000,),
            show_progress_bars=True
        )

        estimator_exponential = HarmonicEvidence()
        estimator_exponential.from_nde(
            all_npe[2], all_nle[2], x=obs,
            shape=(10_000,),
            show_progress_bars=True
        )

        K_est, stdK_est = estimator_zipf.get_bayes_factor(estimator_zipf_mandel)
        print(f"Zipf vs Zipf Mandelbrot: Predicted Bayes Factor: {K_est:.5f} +/- {stdK_est:.5f}")

        K_est, stdK_est = estimator_zipf.get_bayes_factor(estimator_exponential)
        print(f"Zipf vs Exponential: Predicted Bayes Factor: {K_est:.5f} +/- {stdK_est:.5f}")

    else:
        raise ValueError('Unknown evidence estimator')

    return


def evaluate_models(obs, output_dir, run_name, all_model):

    # alpha = 3
    # obs = generate_from_zipf(1000, alpha)

    compressed_obs = create_compressed_obs(obs)
    print(compressed_obs)

    def zipf(x, alpha):
        return 1 / np.power(x, alpha) / np.sum(1 / np.power(rank, alpha))
    
    def zipf_mandel(x, alpha, b):
        return 1 / np.power(x+b, alpha) / np.sum(1 / np.power(rank+b, alpha))
    
    def exponential(x, beta):
        return np.exp(- beta * x)

    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    rank = np.arange(1, len(obs)+1)
    ax.set_title(run_name[0].upper() + run_name[1:])
    ax.plot(rank, obs, 'ko', label='Observed')

    for name in all_model:

        with open(f'{output_dir}/{name}_npe/posterior_ensemble.pkl', 'rb') as f:
            posterior_ensemble = pickle.load(f)

        if name == 'zipf':
            labels = [r'$\alpha$']
        elif name == 'zipf_mandel':
            labels = [r'$\alpha$', r'$b$']
        elif name == 'exponential':
            labels = [r'$\beta$']
        else:
            raise ValueError('Unknown simulator')
        
        metric = PlotSinglePosterior(
            num_samples=5000, sample_method='direct', 
            labels=labels,
            save_samples=True,
            out_dir=output_dir,
        )
        fig2 = metric(
            posterior=posterior_ensemble,
            x_obs = compressed_obs,
            signature=name + '_',
        )
        plt.close(fig2.fig)

        # Plot the fits on top of the data
        samples = np.load(f'{output_dir}/{name}_single_samples.npy')
        if name == 'zipf':
            samples[:, 0] = 1 / samples[:, 0]
            fits = np.array([zipf(rank, *s) for s in samples])
        elif name == 'zipf_mandel':
            samples[:, 0] = 1 / samples[:, 0]
            samples[:, 1] = 1 / samples[:, 1]
            fits = np.array([zipf_mandel(rank, *s) for s in samples])
        elif name == 'exponential':
            fits = np.array([exponential(rank, *s) for s in samples])
        else:
            raise ValueError('Unknown simulator')
        
        # Normalise the fits
        fits = fits / np.sum(fits, axis=1)[:, None] * obs.sum()

        mean_fit = np.mean(fits, axis=0)
        upper_fit = np.percentile(fits, 84, axis=0)
        lower_fit = np.percentile(fits, 16, axis=0)
        label = name
        label = list(label)
        for i, l in enumerate(label):
            if l == '_':
                label[i+1] = label[i+1].upper()
        label[0] = label[0].upper()
        label = ''.join(label)
        label = label.replace('_', '-')
        ax.plot(rank, mean_fit, label=label)
        ax.fill_between(rank, lower_fit, upper_fit, alpha=0.3)
        
    ax.set_ylim(max(0.8, ax.get_ylim()[0]), None)
    ax.set_yscale('log')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{run_name}_fits.png')
    plt.show()

    print('\n' + '-'*50)
    print('Evidence evaluation')
    print('-'*50 + '\n')

    # Get evidence
    for name in all_model[:-1]:
        with open(f'{output_dir}/{name}_{all_model[-1]}_evidence_network.pkl', 'rb') as f:
            runner = pickle.load(f)
        logK = runner.predict(compressed_obs).detach().numpy()[0]
        print(f"Zipf vs {name}: Predicted Bayes Factor (positive prefers {all_model[-1]}):")
        print(f'Predicted lnK: {-logK[0]}')

    return


def simple_fitting_plot():

    obs = generate_from_zipf(1000, 3)

    alpha_fit, _ = fit_to_zipf(obs)
    alpha_poisson = poisson_fit(obs)
    print('True alpha:', alpha)
    print('Fitted alpha:', alpha_fit)
    print('Poisson alpha:', alpha_poisson)
    rank = np.arange(1, len(obs)+1)
    plt.plot(rank, obs, 'ro', label='Observed distribution')
    plt.plot(rank, np.sum(obs) / np.power(rank, alpha) / np.sum(1 / np.power(rank, alpha)), 'b-', label='True distribution')
    plt.plot(rank, np.sum(obs) / np.power(rank, alpha_fit) / np.sum(1 / np.power(rank, alpha_fit)), 'g-', label='Fitted distribution')
    plt.plot(rank, np.sum(obs) / np.power(rank, alpha_poisson) / np.sum(1 / np.power(rank, alpha_poisson)), 'k-', label='Poisson fit')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('fit_zipf.png')
    plt.show()

    return


def make_data():

    inflation_data = [
    1, 110/181, 84/181, 167/362, 81/362, 35/181, 59/362,
    57/362, 45/362, 31/362, 13/181, 25/362, 11/181, 17/362,
    9/362, 7/362, 3/181, 3/362, 1/181, 1/181, 1/362, 1/362, 1/362]
    inflation_data = (np.array(inflation_data) * 362).astype(int)
    np.savetxt('data/inflation_data.txt', inflation_data, fmt='%i')
    
    feynman_data = [
        350/1173, 5/23, 2/17, 116/1173, 97/1173, 79/1173, 12/391,
        10/391, 6/391, 11/1173, 11/1173, 10/1173, 3/391, 5/1173,
        2/1173, 2/1173, 2/1173, 1/1173, 1/1173]
    feynman_data = (np.array(feynman_data) * 1173).astype(int)
    np.savetxt('data/feynman_data.txt', feynman_data, fmt='%i')
        
    wiki_data = [
    169/550, 47/275, 37/275, 3/25, 18/275, 1/22,
    1/22, 17/550, 8/275, 9/550, 4/275, 1/110, 2/275,
    1/550, 1/550]
    wiki_data = (np.array(wiki_data) * 550).astype(int)
    np.savetxt('data/wiki_data.txt', wiki_data, fmt='%i')
    
    return


def main():

    N = 1000  # number of words in corpus
    Nsim = 10_000  # number of simulations to generate

    make_data()

    all_run_name = ['inflation', 'feynman', 'wiki']

    for run_name in all_run_name:
        obs = np.loadtxt(f'data/{run_name}_data.txt').astype(int)
        N = obs.sum()
        
        print('\n' + '-'*50)
        print('\nTraining', run_name, obs.shape, N)
        print('\n' + '-'*50 + '\n')
        outdir = f'results/{run_name}'

        to_train = ['zipf', 'zipf_mandel', 'exponential']
        all_nle, all_npe, all_loader = train(N, Nsim, outdir, to_train=to_train, test_size=0.1)
        train_evidence(N, all_nle, all_npe, all_loader, outdir, evidence_estimator='evidence_network')
        
        evaluate_models(obs, outdir, run_name, ['zipf', 'zipf_mandel', 'exponential'])

    return


if __name__ == '__main__':
    main()

