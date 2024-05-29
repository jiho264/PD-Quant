quant/block_recon.py
    - LossFunction:
        - pd_loss = JensenShannonDivloss(output, output_fp)
quant/layer_recon.py
    - LossFunction
        - pd_loss = JensenShannonDivloss(output, output_fp)


lamb_r = 0.05 : 68.89%
lamb_r = 0.1 : 69.02% 
lamb_r = 0.2 : 68.86%
lamb_r = 0.5 : 68.90%
lamb_r = 1.0 : 68.83%


(1st case) : lamb_r == 0.1


def JensenShannonDivLoss(self, output, output_fp):
        """
        24.05.28 @Lee
        It was inspired by https://arxiv.org/pdf/2109.03228 (3.1.2 Probability Loyalty)
        They applied JS-div loss.

        Args:
            output (tensor): The output of quantized model.
            output_fp (tensor): The output of original FP model.

        Returns:
            js_loss (tensor): The loss of JSDiv(output, output_fp)
        """
        # Calculate softmax outputs
        softmax_output = F.softmax(output / self.T, dim=1)
        softmax_output_fp = F.softmax(output_fp / self.T, dim=1)

        # Calculate average probabilities
        avg_p = 0.5 * (softmax_output + softmax_output_fp)

        # Calculate KL divergence for both directions
        kl_div_1 = F.kl_div(
            F.log_softmax(output / self.T, dim=1), avg_p, reduction="batchmean"
        )
        kl_div_2 = F.kl_div(
            F.log_softmax(output_fp / self.T, dim=1), avg_p, reduction="batchmean"
        )

        # Calculate Jensen-Shannon divergence
        js_div = 0.5 * (kl_div_1 + kl_div_2)

        # Apply regularization
        js_loss = js_div / self.lam

        return js_loss
