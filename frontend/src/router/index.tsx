import React, { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

interface RouterContextType {
  path: string;
  navigate: (to: string) => void;
}

const RouterContext = createContext<RouterContextType>({
  path: '/',
  navigate: () => {},
});

export const useRouter = (): RouterContextType => useContext(RouterContext);

export interface RouterProps {
  children: ReactNode;
}

export const VALID_ROUTES = ['/', '/app', '/research', '/lab', '/results', '/artifacts', '/future'] as const;
export type ValidRoute = typeof VALID_ROUTES[number];

const normalizeRoute = (rawPath: string): string => {
  const trimmed = rawPath.replace(/\/+$/, '') || '/';
  return (VALID_ROUTES as readonly string[]).includes(trimmed) ? trimmed : '/';
};

export const Router: React.FC<RouterProps> = ({ children }) => {
  const [path, setPath] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return normalizeRoute(window.location.pathname);
    }
    return '/';
  });

  useEffect(() => {
    const handlePopState = () => {
      setPath(normalizeRoute(window.location.pathname));
    };

    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  const navigate = (to: string) => {
    const normalized = normalizeRoute(to);
    if (normalized === path && window.location.pathname === normalized) return;
    window.history.pushState({}, '', normalized);
    setPath(normalized);
    window.scrollTo({ top: 0, behavior: 'instant' });
  };

  return (
    <RouterContext.Provider value={{ path, navigate }}>
      {children}
    </RouterContext.Provider>
  );
};

export interface RouteProps {
  path: string;
  element: ReactNode;
}

export const Route: React.FC<RouteProps> = ({ path: routePath, element }) => {
  const { path } = useRouter();
  if (path !== routePath) return null;
  return <>{element}</>;
};

export interface LinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  to: string;
  children: ReactNode;
  className?: string;
}

export const Link: React.FC<LinkProps> = ({ to, children, className = '', onClick, ...props }) => {
  const { navigate } = useRouter();

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (onClick) onClick(e);
    if (!e.defaultPrevented && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey && e.button === 0) {
      e.preventDefault();
      navigate(to);
    }
  };

  return (
    <a href={to} onClick={handleClick} className={className} {...props}>
      {children}
    </a>
  );
};
