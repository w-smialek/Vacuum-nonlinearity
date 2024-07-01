W przypadku przestrzennie jednorodnego pola elektrycznego spolaryzowanego liniowo, wyniki porównałem raz jeszcze z *Kamiński, Krajewska (2019) "Unitary versus pseudounitary ..."*, konkretnie z fig. 5. Wyniki mojej symulacji pokrywają się z artykułem na tyle dokładnie, na ile można to sprawdzić bez danych liczbowych. 

![](./plots/porownanie.png)

Wykonałem symulacje dwuwymiarowego rozkładu pędów liniowo spolaryzowanego światła dla parametrów i postaci pola z artykułu *Bechler, Velez, Krajewska, Kamiński (2023)*. Tutaj wykresy z fig. 2. tego artykułu dla elektronów i moje wyniki dla spinu 0. Dla tych parametrów w przypadku spinu 0 nie występują żadne wiry ani nie zachodzi łączenie się ich.

![](./plots/vortices.png)

W końcu, dla bozonów po niewielkiej modyfikacji można tym samym sposobem symulować efekt działania pola o polaryzacji zależnej od czasu. Przyjąłem następującą postać pola:
$$\vec{\mathcal{E}}(t) = \begin{cases} \mathcal{E}_0 \begin{pmatrix}\sin^2(n_{\text{rep}}\omega t/2)\cos(n_{\text{rep}}n_{\text{osc}}\omega t+\chi)\cos(\delta) \\ \sigma \sin^2(n_{\text{rep}}\omega t/2)\sin(n_{\text{rep}}n_{\text{osc}}\omega t+\chi)\sin(\delta) \end{pmatrix} \ \text{if} \ \ 0 < \omega t < 2\pi n_\text{rep} \\ 0 \qquad \text{else} \end{cases} $$ 
Gdzie $n_{\text{rep}}$ to ilość powtórzeń impulsu, a $n_{\text{osc}}$ to ilość oscylacji w powtórzeniu.
Pole o dowolnej polaryzacji zależnej od czasu można przedstawić na wykresie we współrzędnych biegunowych $r(\phi) = |\vec{\mathcal{E}}| \left(\arg(\vec{\mathcal{E}})\right)$. Poniżej wykres przebiegu impulsu i rozkład pędów dla parametrów:
$$ \mathcal{E}_0 = 0.5  \frac{m_e^2 c^3}{e}, \ \omega = 1.2 m_ec^2, \ n_\text{rep} = 3, \ n_\text{osc}=2, \ \chi = \frac{\pi}{2}, \ \sigma = 1, \ \delta = \frac{\pi}{8}$$
Przyjmuję tutaj konwencję z artykułu *Kamiński, Krajewska (2019)*, gdzie w analitycznych wyrażeniach $\hbar = 1$, natomiast w symulacjach numerycznych $m = m_e = c = h = 1$.
Dodatkowo, na poniższym wykresie $e=1$, tak że bezwymiarowa amplituda oznacza ułamek krytycznego natężenia pola Schwingera $\mathcal{E}_S = \frac{m_e^2 c^3}{e}$.

![](./plots/polar.png)

Amplituda rozkładu pędów:

![](./plots/2D/2D_test_amp.png)
Faza rozkładu pędów:
![](./plots/2D/2D_test_ang.png)
Analogiczne wykresy dla zmodyfikowanych parametrów $\mathcal{E}_0 = 0.1 \mathcal{E}_S, \ \omega = m_e c^2, \  \sigma = 0.3$ (pozostałe parametry pozostawione bez zmian):

![](./plots/polar1.png)

![](./plots/2D/2D_test_amp2.png)

![](./plots/2D/2D_test_ang2.png)

The next step will be improving code efficiency by implementing differential equation solver in cython or JIT compiler Numba, so that the resolution can be significantly enhanced and different field parameters can be studied in a reasonable execution time.
