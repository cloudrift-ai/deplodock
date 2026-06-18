import React from 'react';

export default function Home() {
    React.useEffect(() => {
        window.location.href = 'getting_started';
    }, []);
    return null;
}
