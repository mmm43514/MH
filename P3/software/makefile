# Directorios
BIN=./bin
INC=./include
SRC=./fuentes

# Opciones de compilación
# -Wall para todas las advertencias
# -O2 para optimizacion
OPT=-Wall -O2 -I./$(INC)


all: $(BIN)/p3

# Compilamos
$(BIN)/p3:
	g++ $(OPT) $(SRC)/p3.cpp -o $(BIN)/p3


# Limpieza
clean:
	rm -f $(BIN)/p3
