class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierdo = None
        self.derecho = None


class ArbolBinario:
    def __init__(self):
        self.raiz = None

    # ── Insertar ──────────────────────────────────────────
    def insertar(self, valor):
        if self.raiz is None:
            self.raiz = Nodo(valor)
        else:
            self._insertar_recursivo(self.raiz, valor)

    def _insertar_recursivo(self, nodo, valor):
        if valor < nodo.valor:
            if nodo.izquierdo is None:
                nodo.izquierdo = Nodo(valor)
            else:
                self._insertar_recursivo(nodo.izquierdo, valor)
        else:
            if nodo.derecho is None:
                nodo.derecho = Nodo(valor)
            else:
                self._insertar_recursivo(nodo.derecho, valor)

    # ── Buscar ────────────────────────────────────────────
    def buscar(self, valor):
        return self._buscar_recursivo(self.raiz, valor)

    def _buscar_recursivo(self, nodo, valor):
        if nodo is None:
            return False
        if nodo.valor == valor:
            return True
        if valor < nodo.valor:
            return self._buscar_recursivo(nodo.izquierdo, valor)
        return self._buscar_recursivo(nodo.derecho, valor)

    # ── Eliminar ──────────────────────────────────────────
    def eliminar(self, valor):
        self.raiz = self._eliminar_recursivo(self.raiz, valor)

    def _eliminar_recursivo(self, nodo, valor):
        if nodo is None:
            return None
        if valor < nodo.valor:
            nodo.izquierdo = self._eliminar_recursivo(nodo.izquierdo, valor)
        elif valor > nodo.valor:
            nodo.derecho = self._eliminar_recursivo(nodo.derecho, valor)
        else:
            # Caso 1: hoja
            if nodo.izquierdo is None and nodo.derecho is None:
                return None
            # Caso 2: un solo hijo
            if nodo.izquierdo is None:
                return nodo.derecho
            if nodo.derecho is None:
                return nodo.izquierdo
            # Caso 3: dos hijos → sucesor inorden (mínimo del subárbol derecho)
            sucesor = self._minimo(nodo.derecho)
            nodo.valor = sucesor.valor
            nodo.derecho = self._eliminar_recursivo(nodo.derecho, sucesor.valor)
        return nodo

    def _minimo(self, nodo):
        while nodo.izquierdo:
            nodo = nodo.izquierdo
        return nodo

    # ── Recorridos ────────────────────────────────────────
    def inorden(self):          # izq → raíz → der  (ordenado)
        resultado = []
        self._inorden(self.raiz, resultado)
        return resultado

    def _inorden(self, nodo, resultado):
        if nodo:
            self._inorden(nodo.izquierdo, resultado)
            resultado.append(nodo.valor)
            self._inorden(nodo.derecho, resultado)

    def preorden(self):         # raíz → izq → der
        resultado = []
        self._preorden(self.raiz, resultado)
        return resultado

    def _preorden(self, nodo, resultado):
        if nodo:
            resultado.append(nodo.valor)
            self._preorden(nodo.izquierdo, resultado)
            self._preorden(nodo.derecho, resultado)

    def postorden(self):        # izq → der → raíz
        resultado = []
        self._postorden(self.raiz, resultado)
        return resultado

    def _postorden(self, nodo, resultado):
        if nodo:
            self._postorden(nodo.izquierdo, resultado)
            self._postorden(nodo.derecho, resultado)
            resultado.append(nodo.valor)

    # ── Altura ────────────────────────────────────────────
    def altura(self):
        return self._altura(self.raiz)

    def _altura(self, nodo):
        if nodo is None:
            return 0
        return 1 + max(self._altura(nodo.izquierdo), self._altura(nodo.derecho))


# ── Ejemplo de uso ────────────────────────────────────────
if __name__ == "__main__":
    arbol = ArbolBinario()

    for val in [50, 30, 70, 20, 40, 60, 80]:
        arbol.insertar(val)

    print("Inorden    :", arbol.inorden())    # [20, 30, 40, 50, 60, 70, 80]
    print("Preorden   :", arbol.preorden())   # [50, 30, 20, 40, 70, 60, 80]
    print("Postorden  :", arbol.postorden())  # [20, 40, 30, 60, 80, 70, 50]
    print("Altura     :", arbol.altura())     # 3

    print("¿Existe 40?:", arbol.buscar(40))   # True
    print("¿Existe 99?:", arbol.buscar(99))   # False

    arbol.eliminar(30)
    print("Inorden tras eliminar 30:", arbol.inorden())  # [20, 40, 50, 60, 70, 80]
