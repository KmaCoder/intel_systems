# Python 3.6
import lab01.lexer as lexer


class Term:
    pass


# In App, function names are always considered to be constants, not variables.
# This simplifies things and doesn't affect expressivity. We can always model
# variable functions by envisioning an apply(FUNCNAME, ... args ...).
class App(Term):
    def __init__(self, fname, args=()):
        self.fname = fname
        self.args = args

    def __str__(self):
        return '{0}({1})'.format(self.fname, ','.join(map(str, self.args)))

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.fname == other.fname and
                all(self.args[i] == other.args[i] for i in range(len(self.args))))

    def __len__(self):
        return len(self.__str__())

    __repr__ = __str__


class Var(Term):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

    __repr__ = __str__


class Const(Term):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    __repr__ = __str__


class ParseError(Exception): pass


def parse_term(s):
    """Parses a term from string s, returns a Term."""
    parser = TermParser(s)
    return parser.parse_term()


class TermParser:
    """Term parser.
    Use the top-level parse_term() instead of instantiating this class directly.
    """

    def __init__(self, text):
        self.text = text
        self.cur_token = None
        lexrules = (
            ('\d+', 'NUMBER'),
            ('[a-zA-Z_]\w*', 'ID'),
            (',', 'COMMA'),
            ('\(', 'LP'),
            ('\)', 'RP'),
        )
        self.lexer = lexer.Lexer(lexrules, skip_whitespace=True)
        self.lexer.input(text)
        self._get_next_token()

    def _get_next_token(self):
        try:
            self.cur_token = self.lexer.token()

            if self.cur_token is None:
                self.cur_token = lexer.Token(None, None, None)
        except lexer.LexerError as e:
            self._error('Lexer error at position %d' % e.pos)

    def _error(self, msg):
        raise ParseError(msg)

    def parse_term(self):
        if self.cur_token.type == 'NUMBER':
            term = Const(self.cur_token.val)
            # Consume the current token and return the Const term.
            self._get_next_token()
            return term
        elif self.cur_token.type == 'ID':
            # We have to look at the next token to distinguish between App and
            # Var.
            idtok = self.cur_token
            self._get_next_token()
            if self.cur_token.type == 'LP':
                if idtok.val.isupper():
                    self._error("Function names should be constant")
                self._get_next_token()
                args = []
                while True:
                    args.append(self.parse_term())
                    if self.cur_token.type == 'RP':
                        break
                    elif self.cur_token.type == 'COMMA':
                        # Consume the comma and continue to the next arg
                        self._get_next_token()
                    else:
                        self._error("Expected ',' or ')' in application")
                # Consume the ')'
                self._get_next_token()
                return App(fname=idtok.val, args=args)
            else:
                if idtok.val.isupper():
                    return Var(idtok.val)
                else:
                    return Const(idtok.val)


def occurs_check(v, term, subst):
    """Does the variable v occur anywhere inside term?
    Variables in term are looked up in subst and the check is applied
    recursively.
    """
    assert isinstance(v, Var)
    if v == term:
        return True
    elif isinstance(term, Var) and term.name in subst:
        return occurs_check(v, subst[term.name], subst)
    elif isinstance(term, App):
        return any(occurs_check(v, arg, subst) for arg in term.args)
    else:
        return False


def unify(x, y, subst):
    """Unifies term x and y with initial subst.
    Returns a subst (map of name->term) that unifies x and y, or None if
    they can't be unified. Pass subst={} if no subst are initially
    known. Note that {} means valid (but empty) subst.
    """
    if subst is None:
        return None
    elif x == y:
        return subst
    elif isinstance(x, Var):
        return unify_variable(x, y, subst)
    elif isinstance(y, Var):
        return unify_variable(y, x, subst)
    elif isinstance(x, App) and isinstance(y, App):
        if x.fname != y.fname or len(x.args) != len(y.args):
            return None
        else:
            for i in range(len(x.args)):
                subst = unify(x.args[i], y.args[i], subst)
            return subst
    else:
        return None


def apply_unifier(x, subst):
    """Applies the unifier subst to term x.

    Returns a term where all occurrences of variables bound in subst
    were replaced (recursively); on failure returns None.
    """
    if subst is None:
        return None
    elif len(subst) == 0:
        return x
    elif isinstance(x, Const):
        return x
    elif isinstance(x, Var):
        if x.name in subst:
            return apply_unifier(subst[x.name], subst)
        else:
            return x
    elif isinstance(x, App):
        newargs = [apply_unifier(arg, subst) for arg in x.args]
        return App(x.fname, newargs)
    else:
        return None


def unify_variable(v, x, subst):
    """Unifies variable v with term x, using subst.
    Returns updated subst or None on failure.
    """
    assert isinstance(v, Var)
    if v.name in subst:
        return unify(subst[v.name], x, subst)
    elif isinstance(x, Var) and x.name in subst:
        return unify(v, subst[x.name], subst)
    elif occurs_check(v, x, subst):
        return None
    else:
        # v is not yet in subst and can't simplify x. Extend subst.
        return {**subst, v.name: x}


def create_test_case(size: int) -> (str, str):
    """
    h(x1, x2, …, xn, f(y0, y0), …, f(yn-1, yn-1), yn)
    h(f(x0, x0), f(x1, x1), …, f(xn-1, xn-1), y1, …, yn, xn)
    1. get a string of X's with indices [1, size]
    2.
    """
    xs = [f"X{i + 1}" for i in range(size)]
    xs_str = ",".join(xs)
    fs = [f"f(Y{i},Y{i})" for i in range(size)]
    fs_str = ",".join(fs)
    yn = f"Y{size}"
    h1 = f"h({xs_str},{fs_str},{yn})"

    fs = [f"f(X{i},X{i})" for i in range(size)]
    fs_str = ",".join(fs)
    ys = [f"Y{i + 1}" for i in range(size)]
    ys_str = ",".join(ys)
    xn = f"X{size}"
    h2 = f"h({fs_str},{ys_str},{xn})"
    return h1, h2


if __name__ == '__main__':
    s1, s2 = create_test_case(20)
    # s2 = s2.replace('Y2', 'Y1')
    # s1 = "h(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,f(Y0,Y0),f(Y1,Y1),f(Y2,Y2),f(Y3,Y3),f(Y4,Y4),f(Y5,Y5),f(Y6,Y6),f(Y7,Y7),f(Y8,Y8),f(Y9,Y9),Y10)"
    # s2 = "h(f(X0,X0),f(X1,X1),f(X2,X2),f(X3,X3),f(X4,X4),f(X5,X5),f(X6,X6),f(X7,X7),f(X8,X8),f(X9,X9),Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10,X10)"

    print(s1, s2, sep="\n")

    s1_parsed = parse_term(s1)
    s2_parsed = parse_term(s2)
    subst = unify(s1_parsed, s2_parsed, {})
    print(subst)

    u1 = apply_unifier(s1_parsed, subst)
    u2 = apply_unifier(s2_parsed, subst)
    print(u1 == u2, len(u1))
