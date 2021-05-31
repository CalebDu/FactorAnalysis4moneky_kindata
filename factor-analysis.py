import numpy as np
from scipy import io
import time
import matplotlib.pyplot as plt


class FactorAnalysis:
    def __init__(self, neural: np.ndarray, direction: np.ndarray, n_latent: int = 20, const: float = 0.5,
                 threshold: float = 1):
        # basic parameter
        self.n_latent = n_latent
        self.reach = np.max(direction) + 1
        self.channel = neural.shape[0]
        self.time_span = neural.shape[1]
        self.threshold = threshold

        # load dataset
        self.neural = self.square_root_transform(neural, const).reshape(self.channel, self.time_span, 1).transpose(1, 0,
                                                                                                                   2)
        self.direction = np.squeeze(direction)
        assert self.neural.shape == (self.time_span, self.channel, 1)
        # , f"{self.neural.shape}!= {(self.channel, self.time_span, 1)}"
        assert self.direction.shape == (self.time_span,)

        # initialize gaussian parameter
        self.Pi = self.init_prob()
        self.mu, self.sigma = self.init_gaussian_args()
        self.mapping_mat = np.random.normal(size=(self.channel, self.n_latent)) * 100
        self.R_mat = np.eye(self.channel)
        # self.R_mat = np.diag(np.arange(1, 97))
        assert self.R_mat.shape == (self.channel, self.channel)
        self.print_arg()

    def init_prob(self):
        Pi = np.ones(self.reach)
        for reach in range(self.reach):
            Pi[reach] = np.sum((self.direction == reach)) / self.time_span
        return Pi

    def init_gaussian_args(self):

        mu = np.random.rand(self.reach, self.n_latent, 1) * 100
        sigma = np.random.rand(self.reach, self.n_latent, self.n_latent) * 100
        # mu = np.ones((self.reach, self.n_latent, 1))
        # sigma = np.ones((self.reach, self.n_latent, self.n_latent))
        # for reach in range(self.reach):
        #     mu[reach] = np.mean(self.neural[:, self.direction == reach], axis=1)
        #     sigma[reach] = np.cov(self.neural[:, self.direction == reach].squeeze())
        return mu, sigma

    def E_step(self):
        mu_n = self.mu[self.direction]
        sigma_n = self.sigma[self.direction]

        assert mu_n.shape == (self.time_span, self.n_latent, 1)
        assert sigma_n.shape == (self.time_span, self.n_latent, self.n_latent)
        Rinv = np.linalg.inv(self.R_mat)
        sigma_ninv = np.linalg.pinv(sigma_n)
        Rinv_C = np.einsum("ij,jk->ik", Rinv, self.mapping_mat)
        assert sigma_ninv.shape == (self.time_span, self.n_latent, self.n_latent)

        # A7
        sigma_ninv_plus_CT_Rinv_C = sigma_ninv + np.einsum("ij,jk,kq->iq", self.mapping_mat.T, Rinv, self.mapping_mat)
        assert sigma_ninv_plus_CT_Rinv_C.shape == (self.time_span, self.n_latent, self.n_latent)

        sigma_ninv_plus_CT_Rinv_C_inv = np.linalg.inv(sigma_ninv_plus_CT_Rinv_C)
        assert sigma_ninv_plus_CT_Rinv_C_inv.shape == (self.time_span, self.n_latent, self.n_latent)

        beta_compo = Rinv - np.einsum("ij, ajk, kq->aiq", Rinv_C, sigma_ninv_plus_CT_Rinv_C_inv, Rinv_C.T)
        # beta_compo = self.R_mat + np.einsum("ij,ajk,kg->aig", self.mapping_mat, sigma_n, self.mapping_mat.T)
        assert beta_compo.shape == (self.time_span, self.channel, self.channel)

        beta = np.einsum("aij,jk,akz->aiz", sigma_n, self.mapping_mat.T, beta_compo)
        # beta = np.einsum("aij,jk,akz->aiz", sigma_n, self.mapping_mat.T, np.linalg.inv(beta_compo))
        assert beta.shape == (self.time_span, self.n_latent, self.channel)

        ex_compo = self.neural - np.einsum("ij,ajk->aik", self.mapping_mat, mu_n)
        assert ex_compo.shape == (self.time_span, self.channel, 1)
        # A8
        ex_x = mu_n + np.einsum("aij,ajk->aik", beta, ex_compo)
        assert ex_x.shape == (self.time_span, self.n_latent, 1)
        # A9
        ex_x_xT = sigma_n - np.einsum("aij,jk,akz->aiz", beta, self.mapping_mat, sigma_n) + np.einsum(
            "aij,ajk->aik", ex_x, ex_x.transpose(0, 2, 1))
        assert ex_x_xT.shape == (self.time_span, self.n_latent, self.n_latent)

        neuralT_Rinv_C_x = np.einsum("aij,jk,akq->aiq", np.transpose(self.neural, (0, 2, 1)),
                                     Rinv_C, ex_x).squeeze()
        assert neuralT_Rinv_C_x.shape == (self.time_span,)

        CT_Rinv_C_ex_x_xT = np.einsum("ij,jk, akq->aiq", self.mapping_mat.T, Rinv_C, ex_x_xT)
        assert CT_Rinv_C_ex_x_xT.shape == (self.time_span, self.n_latent, self.n_latent)

        tr_CT_Rinv_C_ex_x_xT = np.trace(CT_Rinv_C_ex_x_xT, axis1=1, axis2=2) * .5
        assert tr_CT_Rinv_C_ex_x_xT.shape == (self.time_span,), f"{tr_CT_Rinv_C_ex_x_xT.shape}"

        mu_nT_sigma_ninv_ex_x = np.einsum("aij,ajk,akq->aiq", np.transpose(mu_n, (0, 2, 1)), sigma_ninv, ex_x).squeeze()
        assert mu_nT_sigma_ninv_ex_x.shape == (self.time_span,)

        sigma_ninv_ex_x_xT = np.einsum("aij,ajk->aik", sigma_ninv, ex_x_xT)
        assert sigma_ninv_ex_x_xT.shape == (self.time_span, self.n_latent, self.n_latent)

        tr_sigma_ninv_ex_x_xT = np.trace(sigma_ninv_ex_x_xT, axis1=1, axis2=2).squeeze() * .5
        assert tr_sigma_ninv_ex_x_xT.shape == (self.time_span,)

        neuralT_Rinv_neural = .5 * np.einsum("aij,jk,akq->aiq", np.transpose(self.neural, (0, 2, 1)), Rinv,
                                             self.neural).squeeze()
        assert neuralT_Rinv_neural.shape == (self.time_span,)

        log_R = 0.5 * np.log(np.linalg.norm(self.R_mat))
        mu_nT_sigma_ninv_mu_n = 0.5 * np.einsum("aij,ajk,akq->aiq", np.transpose(mu_n, (0, 2, 1)), sigma_ninv,
                                                mu_n).squeeze()
        assert mu_nT_sigma_ninv_mu_n.shape == (self.time_span,)

        log_sigma_n = 0.5 * np.log(np.linalg.norm(sigma_n, axis=(1, 2)))
        assert log_sigma_n.shape == (self.time_span,)
        # A13
        likelihood = neuralT_Rinv_C_x - tr_CT_Rinv_C_ex_x_xT + mu_nT_sigma_ninv_ex_x - tr_sigma_ninv_ex_x_xT - neuralT_Rinv_neural - log_R - mu_nT_sigma_ninv_mu_n - log_sigma_n
        assert likelihood.shape == (self.time_span,)

        sum_likelihood = np.sum(likelihood)
        return sum_likelihood, ex_x, ex_x_xT

    def M_step(self, ex_x, ex_x_xT):
        for reach in range(self.reach):
            # A15
            self.mu[reach] = np.mean(ex_x[self.direction == reach], axis=0)
            assert self.mu[reach].shape == (self.n_latent, 1)
            # A16
            self.sigma[reach] = np.mean(ex_x_xT[self.direction == reach], axis=0) - np.einsum("ij,jk->ik",
                                                                                              self.mu[reach],
                                                                                              self.mu[reach].T)
            assert self.sigma[reach].shape == (self.n_latent, self.n_latent)

        neural_ex_xT = np.einsum("aij,ajk->ik", self.neural, np.transpose(ex_x, (0, 2, 1)))
        assert neural_ex_xT.shape == (self.channel, self.n_latent)

        ex_x_xT_inv = np.linalg.inv(np.einsum("aij->ij", ex_x_xT))
        # A17
        self.mapping_mat = np.einsum("ij,jk->ik", neural_ex_xT, ex_x_xT_inv)
        assert self.mapping_mat.shape == (self.channel, self.n_latent)
        neural_neuralT = np.einsum("aij,ajk->ik", self.neural, np.transpose(self.neural, (0, 2, 1)))
        C_ex_x_neuralT = np.einsum("ij,ajk,akq->iq", self.mapping_mat, ex_x, np.transpose(self.neural, (0, 2, 1)))

        # A18
        self.R_mat = np.diag(np.diag(neural_neuralT - C_ex_x_neuralT)) / self.time_span
        assert self.R_mat.shape == (self.channel, self.channel)
        # print(np.diag(self.R_mat))

    def inference(self, new_observation: np.ndarray):
        # A19
        infer_start = time.time()
        time_span = new_observation.shape[1]

        new_observation = new_observation.reshape((-1, 1, self.channel, 1)).repeat(self.reach, axis=1)
        assert new_observation.shape == (time_span, self.reach, self.channel, 1), f"{new_observation.shape}"

        C_sigma_CT_plus_R = self.R_mat + np.einsum("ij,ajk,kq->aiq", self.mapping_mat, self.sigma, self.mapping_mat.T)
        assert C_sigma_CT_plus_R.shape == (self.reach, self.channel, self.channel)

        norm_C_sigma_CT_plus_R = np.linalg.norm(C_sigma_CT_plus_R, axis=(1, 2))
        assert norm_C_sigma_CT_plus_R.shape == (self.reach,)

        reci_norm_C_sigma_CT_plus_R = np.power(norm_C_sigma_CT_plus_R, -0.5)
        assert reci_norm_C_sigma_CT_plus_R.shape == (self.reach,)

        obsv_minus_C_mu = new_observation - np.einsum("ij, ajk->aik", self.mapping_mat, self.mu)[np.newaxis:]
        assert obsv_minus_C_mu.shape == (time_span, self.reach, self.channel, 1)

        C_sigma_CT_plus_Rinv = np.linalg.inv(C_sigma_CT_plus_R)
        assert C_sigma_CT_plus_Rinv.shape == (self.reach, self.channel, self.channel)

        xTAx = -0.5 * np.einsum("abij,bjk,abkq->abiq", np.transpose(obsv_minus_C_mu, (0, 1, 3, 2)),
                                C_sigma_CT_plus_Rinv,
                                obsv_minus_C_mu).squeeze()
        assert xTAx.shape == (time_span, self.reach)

        exp_xTAx = np.exp(xTAx)
        coefficient = np.einsum("i, i->i", self.Pi, reci_norm_C_sigma_CT_plus_R)
        assert coefficient.shape == (self.reach,)

        posterior = np.argmax(np.einsum("ij,j->ij", exp_xTAx, coefficient), axis=1)
        assert posterior.shape == (time_span,)
        factor = np.einsum("i, ij->")
        infer_end = time.time()
        print(f"finished inference in {infer_end - infer_start: .2f}s")
        return predict
        # print(predict)

    def square_root_transform(self, neural, const: float = 0):
        neural.astype(np.float64)
        if const < 0:
            neural[neural > const] = np.sqrt(neural[neural > const] + const)
        else:
            neural = np.sqrt(neural + const)
        return neural

    def fit(self):
        likelihood_history = []
        prev_likelihood, likelihood = np.inf, 0
        idx = 1
        while abs(prev_likelihood - likelihood) > self.threshold and idx <= 20:
            prev_likelihood = likelihood
            e_start = time.time()
            likelihood, ex_x, ex_x_xT = self.E_step()
            e_end = time.time()
            print(f"{idx} iter finished E step in  {e_end - e_start:.2f}s, likelihood: {likelihood:.3f}")
            m_start = time.time()
            self.M_step(ex_x, ex_x_xT)
            m_end = time.time()
            print(f"{idx} iter finished M step in  {m_end - m_start:.2f}s")
            likelihood_history.append(likelihood)
            idx += 1
        plt.plot(range(1, len(likelihood_history) + 1), likelihood_history)
        plt.show()
        return likelihood_history

    def print_arg(self):
        print(f"time_span: {self.time_span}, channel: {self.channel}\nn_latent: {self.n_latent}, n_reach:{self.reach}")


if __name__ == '__main__':
    data = io.loadmat("2019070402_s96.mat")

    index = [23, 69, 90]
    neural_data = data["NeuralData"][:, :3000]
    direction_data = data["DirectionNo"][:, :3000] - 1
    neural_data = np.delete(neural_data, index, axis=0)

    # print(neural_data.shape, direction_data.shape)
    FA = FactorAnalysis(neural_data, direction_data, n_latent=30, const=0.5)
    FA.fit()
    predict = FA.inference(FA.square_root_transform(neural_data, const=0.5))

