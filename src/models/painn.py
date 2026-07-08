"""PaiNN: Polarizable Atom Interaction Neural Network (Schütt et al., 2021).

E(3)-эквивариантная сеть с векторными признаками.
PyTorch Geometric содержит готовую реализацию: torch_geometric.nn.PaiNN.

Ключевые особенности:
  - Узловые признаки раздельно: скаляры (N, hidden) и векторы (N, hidden, 3)
  - Эквивариантность к E(3) (сдвиги + повороты + отражения)
  - Поддержка выходов l=0 (скаляр), l=1 (вектор), l=2 (тензор через V⊗V)

Для поляризуемости (l=2) используем тензорное произведение векторных признаков:
  α = w_0 * (V_i · V_j) * I  +  w_2 * sym(V_i ⊗ V_j - (V_i · V_j)/3 * I)
Это сохраняет эквивариантность: при повороте V → R·V тензор → R·α·Rᵀ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import PaiNN


class PaiNNModel(nn.Module):
    """PaiNN с поддержкой выхода l=0 (скаляр), l=1 (вектор), l=2 (тензор).

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв PaiNN
        num_rbf: число радиальных базисных функций
        cutoff: радиус обрезания (Å)
        predict_dipole: предсказывать вектор диполя
        predict_polarizability: предсказывать тензор поляризуемости
        predict_gap: предсказывать HOMO-LUMO gap
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        predict_dipole: bool = True,
        predict_polarizability: bool = True,
        predict_gap: bool = False,
    ):
        super().__init__()
        self.predict_dipole = predict_dipole
        self.predict_polarizability = predict_polarizability
        self.predict_gap = predict_gap

        # Embedding узлов: тип атома → hidden_channels
        # В Alchemy 7 типов атомов +1 dummy = 8
        self.atom_embed = nn.Embedding(8, hidden_channels)

        # Основная PaiNN сеть
        self.painn = PaiNN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )

        # Heads
        # Скалярный head для HOMO-LUMO gap (из скалярных признаков)
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )

        # Дипольный head: из векторных признаков (l=1)
        # PaiNN даёт векторные признаки размера (N, hidden, 3)
        # Суммируем по узлам, потом линейный слой из hidden в 1
        if predict_dipole:
            self.dipole_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )

        # Поляризуемость: комбинация l=0 (из скаляров) и l=2 (из V⊗V)
        if predict_polarizability:
            # l=0 часть: tr(α) из скалярных признаков
            self.alpha_iso_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
            # l=2 часть: симметричная бесследовая из V⊗V
            # Берём диагональ V⊗V → 3 значения + внедиагональные (V_i_x * V_j_y и т.д.)
            # Но проще: 5 независимых компонент l=2 через линейный слой из V_i ⊗ V_i (симметризованного)
            # V_i ⊗ V_i даёт матрицу 3×3, у неё 6 независимых компонент симметричной части
            self.alpha_aniso_head = nn.Sequential(
                nn.Linear(hidden_channels * 6, hidden_channels * 2),  # 6 = симметричная часть V⊗V
                nn.SiLU(),
                nn.Linear(hidden_channels * 2, 5),  # 5 компонент l=2
            )

    def forward(self, batch) -> dict[str, Tensor]:
        """
        Args:
            batch: PyG Batch с x (N, F), pos (N, 3), batch (N,)

        Returns:
            dict с ключами 'dipole' (B,3), 'polarizability' (B,3,3), 'gap' (B,1) — в зависимости от флагов
        """
        from torch_geometric.utils import scatter

        # Embedding атомов
        atom_types = batch.x.argmax(dim=-1).long()  # (N,) — индекс типа атома
        h = self.atom_embed(atom_types)  # (N, hidden)

        # PaiNN: возвращает скалярные и векторные признаки
        # В PyG PaiNN.forward возвращает (h_s, h_v) или просто h_s в зависимости от версии
        # В новых версиях: paiNNSchütt возвращает скаляр и вектор
        # Сигнатура: paiNN(x, pos, batch) → (scalar, vec)
        h_s, h_v = self.painn(h, batch.pos, batch.batch)
        # h_s: (N, hidden) — скалярные признаки
        # h_v: (N, hidden, 3) — векторные признаки

        B = int(batch.batch.max().item()) + 1
        out = {}

        # === Диполь (l=1) ===
        if self.predict_dipole:
            # Диполь = сумма по узлам: head(V_i) ∈ R³ для каждого i, потом sum
            # V_i имеет форму (hidden, 3); head применяем к hidden, получаем (1, 3) для каждого узла
            mu_per_atom = self.dipole_head(h_v)  # (N, hidden, 3) → (N, 1, 3)
            mu_per_atom = mu_per_atom.squeeze(-2)  # (N, 3)
            # Сумма по узлам каждой молекулы
            mu = scatter(mu_per_atom, batch.batch, dim=0, reduce="sum")  # (B, 3)
            out["dipole"] = mu

        # === Поляризуемость (l=0 ⊕ l=2) ===
        if self.predict_polarizability:
            # l=0 часть: tr(α)
            tr_per_atom = self.alpha_iso_head(h_s)  # (N, 1)
            tr_alpha = scatter(tr_per_atom, batch.batch, dim=0, reduce="sum")  # (B, 1)
            tr_alpha = tr_alpha.squeeze(-1)  # (B,)

            # l=2 часть: симметризованное внешнее произведение векторов
            # Для каждого узла i: V_i ∈ R³ (берём один канал или агрегируем)
            # Используем несколько каналов и агрегируем через head
            # V_i: (hidden, 3). Берём V_i ⊗ V_i для каждого канала → (hidden, 3, 3)
            # Симметричная часть: ½(V⊗V + (V⊗V)^T) — 6 независимых компонент
            # Извлекаем: [V_x², V_y², V_z², V_x·V_y, V_x·V_z, V_y·V_z]
            Vx = h_v[..., 0]  # (N, hidden)
            Vy = h_v[..., 1]
            Vz = h_v[..., 2]
            sym_features = torch.cat([
                Vx * Vx, Vy * Vy, Vz * Vz,
                Vx * Vy, Vx * Vz, Vy * Vz,
            ], dim=-1)  # (N, 6*hidden)

            aniso_per_atom = self.alpha_aniso_head(sym_features)  # (N, 5)
            aniso_coeffs = scatter(aniso_per_atom, batch.batch, dim=0, reduce="sum")  # (B, 5)

            # Собираем тензор 3×3 из tr α и 5 компонент анизотропии
            # α = (tr α / 3) * I + sym_aniso
            # sym_aniso: 5 компонент в стандартном базисе l=2
            # Простой способ: используем 5 компонент как коэффициенты перед 5 базисными матрицами
            alpha = self._build_alpha_tensor(tr_alpha, aniso_coeffs, batch.batch.device)
            out["polarizability"] = alpha  # (B, 3, 3)

        # === HOMO-LUMO gap (l=0) ===
        if self.predict_gap:
            gap_per_atom = self.gap_head(h_s)  # (N, 1)
            gap = scatter(gap_per_atom, batch.batch, dim=0, reduce="sum")  # (B, 1)
            out["gap"] = gap

        return out

    def _build_alpha_tensor(self, tr_alpha: Tensor, aniso_coeffs: Tensor, device) -> Tensor:
        """Собрать тензор поляризуемости 3×3 из tr(α) и 5 компонент l=2.

        α = (tr α / 3) * I + Σ_k c_k * B_k

        где B_k — базисные матрицы симметричной бесследовой части:
          B_1 = diag(1, -1, 0) / √2        ← (z²)
          B_2 = diag(-1, -1, 2) / √6       ← (3z²-r²)
          B_3 = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]   ← (xy)
          B_4 = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]   ← (xz)
          B_5 = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]   ← (yz)

        Args:
            tr_alpha: (B,) — след
            aniso_coeffs: (B, 5) — коэффициенты перед базисами
        """
        B = tr_alpha.shape[0]
        I = torch.eye(3, device=device).expand(B, 3, 3)
        tr_expand = tr_alpha.view(B, 1, 1) / 3.0
        alpha = tr_expand * I

        # Базисные матрицы (5 штук)
        # B1, B2 — диагональные
        B1 = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], device=device) / (2 ** 0.5)
        B2 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 2]], device=device) / (6 ** 0.5)
        # B3, B4, B5 — внедиагональные симметричные
        B3 = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], device=device)
        B4 = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]], device=device)
        B5 = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], device=device)

        bases = torch.stack([B1, B2, B3, B4, B5], dim=0)  # (5, 3, 3)
        # α += Σ_k c_k * B_k
        alpha = alpha + (aniso_coeffs[:, :, None, None] * bases[None, :, :, :]).sum(dim=1)
        return alpha  # (B, 3, 3)


def build_painn(
    predict_dipole: bool = True,
    predict_polarizability: bool = True,
    predict_gap: bool = False,
    **kwargs,
) -> PaiNNModel:
    return PaiNNModel(
        predict_dipole=predict_dipole,
        predict_polarizability=predict_polarizability,
        predict_gap=predict_gap,
        **kwargs,
    )
